#!/usr/bin/env python3
"""
Multi-concept (triple) eval metrics, OrthA Table 2 "Merged" protocol.

For each cascaded triple composite (outputs/multi_concept/<triple>/<scene>/seedS.png):
  - TA  : CLIP cosine(image, scene/style context only — no subjects)
  - IA  : CLIP cosine(face_crop, each concept's training refs), averaged
  - ID  : per-concept ArcFace detection in the composite

Best-of-N seed selection with configurable skip gates.

Usage:
  python scripts/compute_metrics_multi.py --multi_dir outputs/multi_concept/scaffold \\
      --rank_metric id_min --min_per_concept_id 1.0 --min_ia 0.55 --min_ta 0.55
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compute_metrics import (
    load_clip,
    encode_images,
    encode_texts,
    load_face_model,
    collect_ref_images,
    load_images,
    ID_DETECT_COS_THRESHOLD,
    MULTI_SCENE_TA,
    clipscore_from_cosine,
    all_face_embeddings,
    face_crop_ia,
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def joint_generation_prompt(triggers: List[str], scene_ctx: str) -> str:
    """Full multi prompt with TI triggers (OrthA Fig 7 / Mix-of-Show style)."""
    return f"{' & '.join(triggers)}, {scene_ctx}"


def ta_prompt_for_scene(scene: str, triggers: List[str]) -> str:
    ctx = MULTI_SCENE_TA.get(scene, scene)
    return joint_generation_prompt(triggers, ctx)


def final_images(scene_dir: Path) -> List[Path]:
    return sorted(
        p for p in scene_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTS and "_pass" not in p.stem
    )


def _avg(vals):
    v = [x for x in vals if x is not None]
    return round(float(np.mean(v)), 4) if v else None


def _id_min(id_per_concept: Dict[str, Optional[float]]) -> Optional[float]:
    vals = [v for v in id_per_concept.values() if v is not None]
    return float(min(vals)) if vals else None


def _passes_gates(m: dict, args) -> tuple:
    """Return (skipped: bool, reasons: list[str])."""
    reasons = []
    id_pc = m.get("id_per_concept") or {}
    id_min = m.get("id_min")
    id_mean = m.get("id_mean")

    if args.min_per_concept_id is not None:
        for c, v in id_pc.items():
            if v is None or v < args.min_per_concept_id:
                reasons.append(f"id[{c}]<{args.min_per_concept_id}")
                break

    if args.min_id is not None and id_mean is not None and id_mean < args.min_id:
        reasons.append(f"id_mean<{args.min_id}")

    if args.min_ia is not None and m.get("ia") is not None and m["ia"] < args.min_ia:
        reasons.append(f"ia<{args.min_ia}")

    if args.min_ta is not None and m.get("ta") is not None and m["ta"] < args.min_ta:
        reasons.append(f"ta<{args.min_ta}")

    if args.require_id and id_mean is None:
        reasons.append("no_id")

    return bool(reasons), reasons


def rank_key(m: dict, rank_metric: str) -> tuple:
    def g(k, default=-1.0):
        v = m.get(k)
        return v if v is not None else default

    if rank_metric == "composite":
        w_id, w_ia, w_ta = 0.5, 0.3, 0.2
        comp = (w_id * g("id_min") + w_ia * g("ia") + w_ta * g("ta"))
        return (comp, g("id_min"), g("ia"), g("ta"))
    if rank_metric == "id_min":
        return (g("id_min"), g("ia"), g("ta"))
    if rank_metric == "id_mean":
        return (g("id_mean"), g("ia"), g("ta"))
    if rank_metric == "ia":
        return (g("ia"), g("id_min"), g("ta"))
    return (g("ta"), g("id_min"), g("ia"))


def main():
    ap = argparse.ArgumentParser(description="Multi-concept (triple) eval metrics")
    ap.add_argument("--multi_dir", default="outputs/multi_concept")
    ap.add_argument("--registry", default="concept_registry.json")
    ap.add_argument("--datasets_dir", default="Datasets")
    ap.add_argument("--output", default=None)
    ap.add_argument("--clip_model", default="ViT-B-32")
    ap.add_argument("--clip_pretrained", default="openai")
    ap.add_argument("--no_face", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rank_metric", default="id_min",
                    choices=["id_min", "id_mean", "ia", "ta", "composite"])
    ap.add_argument("--min_id", type=float, default=0.67)
    ap.add_argument("--min_per_concept_id", type=float, default=1.0)
    ap.add_argument("--min_ia", type=float, default=0.55)
    ap.add_argument("--min_ta", type=float, default=0.55)
    ap.add_argument("--require_id", action="store_true")
    args = ap.parse_args()

    multi_dir = Path(args.multi_dir)
    datasets_dir = Path(args.datasets_dir)
    out_path = Path(args.output) if args.output else multi_dir / "metrics.json"

    manifest_path = multi_dir / "triples.json"
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    manifest = json.load(open(manifest_path))
    triple_order: Dict[str, List[str]] = manifest.get("order", {})

    registry = json.load(open(args.registry))
    registry_map = {e["concept"]: e for e in registry if e.get("concept")}

    print("Loading CLIP …")
    backend, clip_model, clip_proc, clip_tok = load_clip(
        args.clip_model, args.clip_pretrained, args.device)

    face_app = None
    if not args.no_face:
        print("Loading ArcFace …")
        face_app = load_face_model(args.device)

    ref_clip: Dict[str, object] = {}
    ref_faces: Dict[str, List[np.ndarray]] = {}
    needed = {c for order in triple_order.values() for c in order}
    for concept in needed:
        entry = registry_map.get(concept, {})
        ref_paths = collect_ref_images(entry, datasets_dir)
        ref_imgs = load_images(ref_paths) if ref_paths else []
        if ref_imgs:
            ref_clip[concept] = encode_images(backend, clip_model, clip_proc,
                                              ref_imgs, args.device)
            if face_app is not None:
                embs = []
                for rim in ref_imgs:
                    embs.extend(all_face_embeddings(face_app, rim)[:1])
                ref_faces[concept] = embs

    results: Dict = {}

    for tri_slug, order in tqdm(triple_order.items(), desc="Triples"):
        tri_dir = multi_dir / tri_slug
        if not tri_dir.is_dir():
            continue

        triggers = [registry_map.get(c, {}).get("trigger") or f"<id_{c}>" for c in order]

        scenes_res: Dict[str, dict] = {}
        for scene_dir in sorted(tri_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            scene = scene_dir.name
            imgs_paths = final_images(scene_dir)
            if not imgs_paths:
                continue

            prompt = ta_prompt_for_scene(scene, triggers)
            txt_feat = encode_texts(backend, clip_model, clip_proc, clip_tok,
                                    [prompt], args.device)

            per_seed: Dict[str, dict] = {}
            for img_path in imgs_paths:
                seed_name = img_path.stem
                img = Image.open(img_path).convert("RGB")
                gi = encode_images(backend, clip_model, clip_proc, [img], args.device)
                cos = float((gi @ txt_feat.T).mean().item())
                ta_i = clipscore_from_cosine(cos)

                if face_app is not None and ref_clip:
                    ia_i = face_crop_ia(
                        img, face_app, ref_clip, order, ref_faces,
                        backend, clip_model, clip_proc, args.device,
                    )
                else:
                    ia_i = float((gi @ ref_clip[order[0]].T).mean().item()) if order[0] in ref_clip else None

                id_pc: Dict[str, Optional[float]] = {}
                id_vals = []
                if face_app is not None:
                    faces = all_face_embeddings(face_app, img)
                    for concept in order:
                        refs = ref_faces.get(concept, [])
                        if not refs:
                            id_pc[concept] = None
                            continue
                        if not faces:
                            det = 0.0
                        else:
                            best = max(float(np.dot(gf, rf))
                                       for gf in faces for rf in refs)
                            det = 1.0 if best > ID_DETECT_COS_THRESHOLD else 0.0
                        id_pc[concept] = det
                        id_vals.append(det)

                id_mean_i = float(np.mean(id_vals)) if id_vals else None
                id_min_i = _id_min(id_pc)

                per_seed[seed_name] = {
                    "ta": round(ta_i, 4),
                    "ia": round(ia_i, 4) if ia_i is not None else None,
                    "id_mean": round(id_mean_i, 4) if id_mean_i is not None else None,
                    "id_min": round(id_min_i, 4) if id_min_i is not None else None,
                    "id_per_concept": {k: (round(v, 4) if v is not None else None)
                                       for k, v in id_pc.items()},
                }

            best_seed, best_m = max(per_seed.items(),
                                    key=lambda item: rank_key(item[1], args.rank_metric))
            skipped, skip_reasons = _passes_gates(best_m, args)

            scenes_res[scene] = {
                "selected_seed": best_seed,
                "ta": best_m["ta"],
                "ia": best_m["ia"],
                "id_mean": best_m["id_mean"],
                "id_min": best_m.get("id_min"),
                "id_per_concept": best_m["id_per_concept"],
                "skipped": skipped,
                "skip_reasons": skip_reasons,
                "n_seeds": len(per_seed),
                "per_seed": per_seed,
            }

        if not scenes_res:
            continue

        kept = {k: v for k, v in scenes_res.items() if not v.get("skipped")}
        results[tri_slug] = {
            "order": order,
            "scenes": scenes_res,
            "n_kept": len(kept),
            "n_scenes": len(scenes_res),
            "avg": {
                "ta":      _avg([s["ta"] for s in kept.values()]),
                "ia":      _avg([s["ia"] for s in kept.values()]),
                "id_mean": _avg([s["id_mean"] for s in kept.values()]),
                "id_min":  _avg([s["id_min"] for s in kept.values() if s.get("id_min") is not None]),
            },
        }

    meta = {
        "rank_metric": args.rank_metric,
        "min_id": args.min_id,
        "min_per_concept_id": args.min_per_concept_id,
        "min_ia": args.min_ia,
        "min_ta": args.min_ta,
    }
    payload = {"meta": meta, "triples": results}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {out_path}")

    header = (f"{'Triple':<28} {'Scene':<10} {'Seed':<8} "
              f"{'TA':>7} {'IA':>7} {'ID_min':>7}  {'Status':<7}")
    print("\n" + "=" * len(header))
    print(f"rank={args.rank_metric}  gates: id_mean>={args.min_id} "
          f"per_concept>={args.min_per_concept_id} ia>={args.min_ia} ta>={args.min_ta}")
    print(header)
    print("-" * len(header))
    for tri_slug in sorted(results):
        for scene, s in results[tri_slug]["scenes"].items():
            ta  = f"{s['ta']:.4f}"      if s.get("ta")      is not None else "   —  "
            ia  = f"{s['ia']:.4f}"      if s.get("ia")      is not None else "   —  "
            idm = f"{s['id_min']:.4f}"  if s.get("id_min")  is not None else "   —  "
            status = "SKIP" if s.get("skipped") else "keep"
            print(f"{tri_slug:<28} {scene:<10} {s.get('selected_seed',''):<8} "
                  f"{ta:>7} {ia:>7} {idm:>7}  {status:<7}")
    print("=" * len(header))

    kept_scenes = [s for r in results.values() for s in r["scenes"].values()
                   if not s.get("skipped")]
    n_total = sum(len(r["scenes"]) for r in results.values())
    all_ta = [s["ta"] for s in kept_scenes if s.get("ta") is not None]
    all_ia = [s["ia"] for s in kept_scenes if s.get("ia") is not None]
    all_id = [s["id_mean"] for s in kept_scenes if s.get("id_mean") is not None]
    print(f"{'OVERALL (kept)':<28} {'(merged)':<10} {len(kept_scenes):<8} "
          f"{(_avg(all_ta) or 0):>7.4f} {(_avg(all_ia) or 0):>7.4f} "
          f"{(_avg(all_id) or 0):>8.4f}  {len(kept_scenes)}/{n_total}")
    print("=" * len(header))

    csv_path = out_path.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["triple", "scene", "selected_seed", "skipped", "skip_reasons",
                    "TA", "IA", "ID_mean", "ID_min", "ID_per_concept"])
        for tri_slug in sorted(results):
            for scene, s in results[tri_slug]["scenes"].items():
                w.writerow([
                    tri_slug, scene, s.get("selected_seed", ""),
                    s.get("skipped", ""),
                    ";".join(s.get("skip_reasons") or []),
                    s.get("ta", ""), s.get("ia", ""),
                    s.get("id_mean", ""), s.get("id_min", ""),
                    ";".join(f"{k}={v}" for k, v in
                             (s.get("id_per_concept") or {}).items()),
                ])
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
