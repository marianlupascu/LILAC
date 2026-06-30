#!/usr/bin/env python3
"""Emit a concept-map template for (re)captioning all datasets.

The VLM captioner cannot know a concept's *name/identity* ("this is gosling")
— that is human knowledge. This script scans Datasets/S* and writes a TSV with
one row per dataset, pre-filling the concept name from an existing
metadata.jsonl when present and leaving it BLANK (TODO) when not, so you fill
in the few unknowns by hand before running scripts/run_recaption_all.sh.

Columns (tab-separated):  dataset  concept  type  trigger_token  has_images

The `type` column (person | animal | object) is recorded here and consumed
by build_concept_registry.py as AUTHORITATIVE, so we don't depend on the
caption text to infer it (the recommended 'scene' caption mode intentionally
omits person descriptors, which would otherwise break type inference).

Usage:
  python scripts/make_concept_map.py --datasets_dir Datasets --output concept_map.tsv
  # then edit concept_map.tsv, fill the blank 'concept'/'type' cells, then:
  bash scripts/run_recaption_all.sh
"""
import argparse
import json
import re
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def natural_key(name: str):
    m = re.search(r"(\d+)", name)
    return (int(m.group(1)) if m else 1 << 30, name)


def existing_concept(meta_path: Path):
    if not meta_path.exists():
        return None, None
    try:
        with open(meta_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                return rec.get("concept"), rec.get("trigger_token")
    except Exception:  # noqa: BLE001
        pass
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets_dir", default="Datasets")
    ap.add_argument("--output", default="concept_map.tsv")
    ap.add_argument("--default_trigger", default="ohwx",
                    help="Trigger token written for datasets lacking one.")
    args = ap.parse_args()

    root = Path(args.datasets_dir)
    ds_dirs = sorted(
        [p for p in root.iterdir() if p.is_dir() and re.match(r"^S\d+$", p.name)],
        key=lambda p: natural_key(p.name),
    )

    rows = []
    n_blank = 0
    for ds in ds_dirs:
        n_img = sum(1 for p in ds.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        concept, trigger = existing_concept(ds / "metadata.jsonl")
        if not concept:
            concept = "TODO"
            n_blank += 1
        trigger = trigger or args.default_trigger
        rows.append((ds.as_posix(), concept, "TODO", trigger, str(n_img)))

    with open(args.output, "w") as f:
        f.write("# dataset\tconcept\ttype\ttrigger_token\thas_images\n")
        f.write("# Fill every concept marked TODO with the real concept name "
                "(e.g. 'cat', 'watch', 'einstein') and type with one of "
                "person|animal|object. Rows with has_images=0 are skipped by the "
                "recaptioner. Lines starting with # are ignored.\n")
        for r in rows:
            f.write("\t".join(r) + "\n")

    print(f"Wrote {args.output} ({len(rows)} datasets, {n_blank} need a concept name).")
    if n_blank:
        print(f"  -> Edit {args.output} and replace every 'TODO' concept, then run "
              f"scripts/run_recaption_all.sh")
    else:
        print("  -> All concepts known; you can run scripts/run_recaption_all.sh directly.")


if __name__ == "__main__":
    main()
