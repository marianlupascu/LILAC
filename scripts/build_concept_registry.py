#!/usr/bin/env python3
"""Verify all concept datasets and emit a reviewable training registry.

Scans ``Datasets/S*`` (or a given root) and, for each dataset:
  1. Reads the ``concept`` field from metadata.jsonl.
  2. Counts images, checks every ``file_name`` exists and is a readable image.
  3. Infers the concept TYPE (person | animal | object) and a minimal class
     anchor by keyword-counting the captions.
  4. Proposes ``trigger = <id_concept>`` and a small, GENERIC attribute-anchor
     list. The anchors are deliberately few/generic so the Textual-Inversion
     token carries the identity (the lesson from the gosling/swift fixes:
     over-specified anchors let the LoRA satisfy the prompt with a generic
     archetype and the TI token never binds the identity).

Outputs:
  - ``concept_registry.json``  : one entry per dataset (REVIEW THIS before training)
  - ``verification_report.txt``: per-dataset image counts + missing/corrupt flags
  - a printed summary table

Run on the machine that holds the dataset images:

  python scripts/build_concept_registry.py \
      --datasets_dir Datasets \
      --output concept_registry.json \
      --report verification_report.txt

After it runs, open concept_registry.json and fix any wrong type / anchors,
then launch scripts/run_train_all.sh.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# Keyword vocab for type / gender inference from captions.
PERSON_WORDS = {
    "man", "woman", "men", "women", "person", "people", "boy", "girl",
    "guy", "lady", "male", "female", "he", "she", "his", "her", "him",
    "gentleman", "actor", "actress", "singer", "politician",
}
WOMAN_WORDS = {"woman", "women", "girl", "lady", "female", "she", "her", "actress"}
MAN_WORDS = {"man", "men", "boy", "guy", "male", "he", "his", "him", "gentleman", "actor"}
ANIMAL_WORDS = {
    "dog", "puppy", "cat", "kitten", "horse", "bird", "rabbit", "hamster",
    "pet", "animal", "dachshund", "retriever", "spaniel", "terrier", "poodle",
    "feline", "canine", "kitty",
}


def natural_key(name: str):
    """Sort S1, S2, ..., S10, ..., S21 in numeric order."""
    m = re.search(r"(\d+)", name)
    return (int(m.group(1)) if m else 1 << 30, name)


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z]+", text.lower())


def read_metadata(meta_path: Path) -> List[dict]:
    records = []
    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def infer_type_and_gender(captions: List[str],
                          concept: Optional[str]) -> Tuple[str, Optional[str], Counter]:
    """Return (type, gender_or_None, token_counter).

    type in {person, animal, object}. gender in {man, woman} for persons.

    Heuristic priority (the concept NAME is the strongest signal):
      1. If the concept token is itself a frequent common noun in the captions
         (e.g. "sunglasses", "dog", "cat") it is a class word -> object/animal.
         People concepts are proper names (trump/swift/...) that don't appear
         in the descriptive captions, so this cleanly separates them.
      2. Otherwise fall back to person/animal keyword counts.

    This avoids the classic failure where an object's captions say
    "no other people visible" and the bare person-word count misfires.
    """
    counter: Counter = Counter()
    for cap in captions:
        counter.update(tokenize(cap))

    n = max(1, len(captions))
    concept_tokens = tokenize(concept or "")
    concept_hits = sum(counter[w] for w in concept_tokens)

    woman_hits = sum(counter[w] for w in WOMAN_WORDS)
    man_hits = sum(counter[w] for w in MAN_WORDS)
    person_hits = sum(counter[w] for w in PERSON_WORDS)
    animal_hits = sum(counter[w] for w in ANIMAL_WORDS)

    # 1) concept name appears as a frequent noun -> it's a class (object/animal)
    if concept_tokens and concept_hits >= 0.5 * n and not (set(concept_tokens) & PERSON_WORDS):
        if set(concept_tokens) & ANIMAL_WORDS:
            return "animal", None, counter
        return "object", None, counter

    # 2) keyword-based fallback
    if animal_hits >= 2 and animal_hits > person_hits:
        return "animal", None, counter
    if person_hits >= 3 and person_hits >= animal_hits:
        gender = "woman" if woman_hits > man_hits else "man"
        return "person", gender, counter
    return "object", None, counter


def most_common_class_noun(counter: Counter, concept: str,
                           vocab: set) -> Optional[str]:
    """Pick the most frequent vocab word; fall back to concept-as-words."""
    best = None
    best_n = 0
    for w in vocab:
        if counter[w] > best_n:
            best_n = counter[w]
            best = w
    return best


def propose_anchors(concept: str, ctype: str, gender: Optional[str],
                    counter: Counter) -> Tuple[str, List[str]]:
    """Return (class_noun, anchors). Few + generic on purpose."""
    concept_words = concept.replace("_", " ").strip()
    if ctype == "person":
        gender_noun = f"a {gender}" if gender else "a person"
        anchors = [gender_noun, "adult", "realistic portrait"]
        class_noun = gender or "person"
    elif ctype == "animal":
        cls = most_common_class_noun(counter, concept, ANIMAL_WORDS) or concept_words
        anchors = [f"a {cls}", "an animal", "realistic photo"]
        class_noun = cls
    else:  # object
        cls = concept_words or "object"
        # No indefinite article: many object concepts are plural / uncountable
        # ("sunglasses", "glasses", "headphones") where "a sunglasses" is wrong.
        anchors = [cls, "an object", "product photo"]
        class_noun = cls
    return class_noun, anchors


def verify_images(dataset_dir: Path, records: List[dict],
                  check_readable: bool) -> Dict:
    """Cross-check metadata file_names against on-disk images."""
    meta_files = [r.get("file_name") for r in records if r.get("file_name")]
    on_disk = sorted(
        [p.name for p in dataset_dir.iterdir()
         if p.suffix.lower() in IMAGE_EXTS],
        key=natural_key,
    )

    missing = []
    corrupt = []
    for fn in meta_files:
        fp = dataset_dir / fn
        if not fp.exists():
            missing.append(fn)
            continue
        if check_readable:
            try:
                from PIL import Image
                with Image.open(fp) as im:
                    im.verify()
            except Exception as e:  # noqa: BLE001
                corrupt.append(f"{fn} ({e})")

    not_in_meta = sorted(set(on_disk) - set(meta_files), key=natural_key)

    return {
        "num_meta_records": len(meta_files),
        "num_on_disk": len(on_disk),
        "missing": missing,
        "corrupt": corrupt,
        "not_in_metadata": not_in_meta,
    }


def load_concept_map(path: Path) -> Dict[str, Dict]:
    """Read concept_map.tsv -> {dataset_name: {concept, ctype, gender, trigger}}.

    Columns: dataset  concept  type  trigger_token  has_images
    `type` may be person|man|woman|animal|object ('man'/'woman' imply person
    + gender). This map is treated as AUTHORITATIVE so we don't depend on
    caption text for type (scene-mode captions omit person descriptors).
    """
    out: Dict[str, Dict] = {}
    if not path.is_file():
        return out
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            ds_name = Path(parts[0].strip()).name
            concept = parts[1].strip()
            type_raw = parts[2].strip().lower()
            trigger = parts[3].strip() if len(parts) > 3 else ""
            if not concept or concept.upper() == "TODO":
                continue
            if type_raw in ("man", "woman"):
                ctype, gender = "person", type_raw
            elif type_raw in ("person", "animal", "object"):
                ctype, gender = type_raw, None
            else:
                ctype, gender = "", None  # leave to inference
            out[ds_name] = {"concept": concept, "ctype": ctype,
                            "gender": gender, "trigger": trigger}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets_dir", default="Datasets")
    ap.add_argument("--output", default="concept_registry.json")
    ap.add_argument("--report", default="verification_report.txt")
    ap.add_argument("--concept_map", default="concept_map.tsv",
                    help="Optional TSV (from make_concept_map.py) used as the "
                         "authoritative source of concept name + type, so the "
                         "registry is correct even without captions yet.")
    ap.add_argument("--expect_count", type=int, default=21,
                    help="Warn if the number of datasets found differs.")
    ap.add_argument("--no_readable_check", action="store_true",
                    help="Skip PIL.verify() on every image (faster).")
    args = ap.parse_args()

    cmap = load_concept_map(Path(args.concept_map))
    if cmap:
        print(f"Using concept map: {args.concept_map} ({len(cmap)} entries)\n")

    root = Path(args.datasets_dir)
    if not root.is_dir():
        raise SystemExit(f"Datasets dir not found: {root}")

    ds_dirs = sorted(
        [p for p in root.iterdir() if p.is_dir() and re.match(r"^S\d+$", p.name)],
        key=lambda p: natural_key(p.name),
    )
    if not ds_dirs:
        raise SystemExit(f"No S<N> dataset directories under {root}")

    registry = []
    report_lines = []
    seen_concepts: Dict[str, str] = {}
    summary_rows = []

    for ds in ds_dirs:
        meta_path = ds / "metadata.jsonl"
        entry: Dict = {"dataset": ds.name, "dataset_dir": ds.as_posix()}

        mapped = cmap.get(ds.name)

        if not meta_path.exists():
            # No metadata yet. Count on-disk images so the user knows what's
            # recoverable. If the concept map names this dataset, we can still
            # emit a COMPLETE entry (concept + type + anchors) — training's
            # regen_metadata_ti.py builds metadata from the bare images.
            on_disk = sorted(
                [p.name for p in ds.iterdir() if p.suffix.lower() in IMAGE_EXTS],
                key=natural_key,
            )
            n_img = len(on_disk)
            if mapped and n_img > 0:
                concept = mapped["concept"]
                ctype = mapped["ctype"] or "object"
                gender = mapped["gender"]
                class_noun, anchors = propose_anchors(concept, ctype, gender, Counter())
                trigger = mapped["trigger"] or f"<id_{concept}>"
                if not trigger.startswith("<id_"):
                    trigger = f"<id_{concept}>"
                entry.update({
                    "concept": concept, "type": ctype, "gender": gender,
                    "class_noun": class_noun, "trigger": trigger,
                    "attribute_anchors": anchors, "num_images": n_img,
                    "num_metadata_records": 0,
                    "_note": "from concept_map; metadata will be generated at train time",
                })
                if concept in seen_concepts:
                    entry["_review"] = f"DUPLICATE concept '{concept}' (also {seen_concepts[concept]})"
                seen_concepts[concept] = ds.name
                registry.append(entry)
                report_lines.append(
                    f"[{ds.name}] concept={concept!r} type={ctype} gender={gender} "
                    f"(from map) images(disk={n_img}, meta=0)  trigger={trigger} anchors={anchors}")
                summary_rows.append((ds.name, concept, ctype, n_img, "from map (no metadata yet)"))
                continue
            if n_img > 0:
                note = "NEEDS CAPTION (images present, no metadata)"
                warn = "no metadata.jsonl but images present — needs captioning or a concept_map entry"
            else:
                note = "EMPTY (no metadata, no images)"
                warn = "no metadata.jsonl and no images on disk"
            entry.update({"concept": None, "type": "unknown", "trigger": None,
                          "attribute_anchors": [], "num_images": n_img,
                          "num_metadata_records": 0, "_warning": warn})
            registry.append(entry)
            report_lines.append(f"[{ds.name}] {warn} (images on disk={n_img})")
            summary_rows.append((ds.name, "?", "unknown", n_img, note))
            continue

        records = read_metadata(meta_path)
        concept = None
        for r in records:
            if r.get("concept"):
                concept = r["concept"]
                break
        captions = [r.get("raw_caption") or r.get("prompt") or "" for r in records]

        ctype, gender, counter = infer_type_and_gender(captions, concept)
        # Concept map overrides inference when present (authoritative).
        if mapped:
            concept = mapped["concept"] or concept
            if mapped["ctype"]:
                ctype = mapped["ctype"]
            if mapped["gender"] is not None:
                gender = mapped["gender"]
        class_noun, anchors = propose_anchors(concept or ds.name, ctype, gender, counter)
        trigger = f"<id_{concept}>" if concept else f"<id_{ds.name.lower()}>"

        vinfo = verify_images(ds, records, check_readable=not args.no_readable_check)

        flags = []
        if concept and concept in seen_concepts:
            flags.append(f"DUPLICATE concept '{concept}' (also {seen_concepts[concept]})")
        if concept:
            seen_concepts[concept] = ds.name
        if vinfo["missing"]:
            flags.append(f"{len(vinfo['missing'])} missing image(s)")
        if vinfo["corrupt"]:
            flags.append(f"{len(vinfo['corrupt'])} corrupt image(s)")
        if vinfo["num_meta_records"] == 0:
            flags.append("0 metadata records")

        entry.update({
            "concept": concept,
            "type": ctype,
            "gender": gender,
            "class_noun": class_noun,
            "trigger": trigger,
            "attribute_anchors": anchors,
            "num_images": vinfo["num_on_disk"],
            "num_metadata_records": vinfo["num_meta_records"],
        })
        if flags:
            entry["_review"] = "; ".join(flags)
        registry.append(entry)

        report_lines.append(
            f"[{ds.name}] concept={concept!r} type={ctype} gender={gender} "
            f"images(disk={vinfo['num_on_disk']}, meta={vinfo['num_meta_records']})"
        )
        if vinfo["missing"]:
            report_lines.append(f"    MISSING ({len(vinfo['missing'])}): "
                                f"{vinfo['missing'][:10]}")
        if vinfo["corrupt"]:
            report_lines.append(f"    CORRUPT ({len(vinfo['corrupt'])}): "
                                f"{vinfo['corrupt'][:10]}")
        if vinfo["not_in_metadata"]:
            report_lines.append(f"    on-disk not in metadata "
                                f"({len(vinfo['not_in_metadata'])}): "
                                f"{vinfo['not_in_metadata'][:10]}")
        report_lines.append(f"    trigger={trigger}  anchors={anchors}")
        summary_rows.append((ds.name, str(concept), ctype,
                             vinfo["num_on_disk"], "; ".join(flags) or "ok"))

    # Write outputs
    with open(args.output, "w") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    header = (f"Concept registry verification — {len(ds_dirs)} dataset(s) "
              f"under {root}\n" + "=" * 70)
    count_warn = ""
    if len(ds_dirs) != args.expect_count:
        count_warn = (f"\nWARNING: found {len(ds_dirs)} datasets, "
                      f"expected {args.expect_count}.")
    with open(args.report, "w") as f:
        f.write(header + count_warn + "\n\n")
        f.write("\n".join(report_lines) + "\n")

    # Printed summary table
    print(header + count_warn)
    print(f"\n{'dataset':<8} {'concept':<16} {'type':<8} {'imgs':>5}  flags")
    print("-" * 70)
    for name, concept, ctype, n, flags in summary_rows:
        print(f"{name:<8} {concept:<16} {ctype:<8} {n:>5}  {flags}")
    print("-" * 70)
    n_person = sum(1 for r in registry if r.get("type") == "person")
    n_animal = sum(1 for r in registry if r.get("type") == "animal")
    n_object = sum(1 for r in registry if r.get("type") == "object")
    n_flag = sum(1 for r in registry if r.get("_review") or r.get("_warning"))
    print(f"\nTotals: {len(registry)} datasets — "
          f"{n_person} person, {n_animal} animal, {n_object} object "
          f"({n_flag} need review)")
    print(f"\nWrote registry : {args.output}")
    print(f"Wrote report   : {args.report}")
    print("\nNEXT: review/edit anchors + types in the registry, then run:")
    print("  bash scripts/run_train_all.sh")


if __name__ == "__main__":
    main()
