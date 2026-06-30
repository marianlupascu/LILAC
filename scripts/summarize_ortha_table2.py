#!/usr/bin/env python3
"""
OrthA Table 2 summary: Single (t2i, matched scenes) vs Multi methods.

Multi methods (each read from {multi_root}/{method}/metrics.json):
  - ortha    : same-backbone OrthA merge baseline (orthogonal LoRAs, single pass)
  - cascade  : LILAC additive cascade
  - scaffold : LILAC scaffold-and-replace

Δ = M − S  (OrthA Table 2 convention)

Usage:
  python scripts/summarize_ortha_table2.py
  python scripts/summarize_ortha_table2.py \\
      --single_metrics outputs/eval_infer/metrics.json \\
      --multi_root outputs/multi_concept \\
      --methods ortha,cascade,scaffold \\
      --id_threshold 0.5
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compute_metrics import MATCHED_SINGLE_SLUG

MATCHED_SLUGS = ("portrait", "poker", "cyberpunk")


def _avg(vals: List[float]) -> Optional[float]:
    v = [x for x in vals if x is not None]
    return round(sum(v) / len(v), 4) if v else None


def load_multi_metrics(path: Path) -> Dict:
    if not path.exists():
        return {}
    data = json.load(open(path))
    if "triples" in data:
        return data["triples"]
    return data


def successful_concepts(single: Dict, id_threshold: float) -> List[str]:
    out = []
    for concept, cr in single.items():
        t2i = cr.get("t2i", {}) or {}
        id_score = (t2i.get("avg", {}) or {}).get("id")
        if id_score is not None and id_score > id_threshold:
            out.append(concept)
    return sorted(out)


def single_macro(single: Dict, concepts: List[str], slugs: tuple) -> Dict[str, Optional[float]]:
    ta, ia, id_ = [], [], []
    for c in concepts:
        t2i = single.get(c, {}).get("t2i", {})
        per_slug = t2i.get("per_slug", {})
        for slug in slugs:
            if slug not in per_slug:
                continue
            row = per_slug[slug]
            if row.get("ta") is not None:
                ta.append(row["ta"])
            if row.get("ia") is not None:
                ia.append(row["ia"])
            if row.get("id") is not None:
                id_.append(row["id"])
    return {"ta": _avg(ta), "ia": _avg(ia), "id": _avg(id_)}


def multi_macro(multi: Dict) -> Dict[str, Optional[float]]:
    kept = []
    for tri in multi.values():
        for scene, s in tri.get("scenes", {}).items():
            if s.get("skipped"):
                continue
            kept.append(s)
    return {
        "ta": _avg([s["ta"] for s in kept if s.get("ta") is not None]),
        "ia": _avg([s["ia"] for s in kept if s.get("ia") is not None]),
        "id": _avg([s["id_mean"] for s in kept if s.get("id_mean") is not None]),
        "n_kept": len(kept),
        "n_total": sum(len(t.get("scenes", {})) for t in multi.values()),
    }


def fmt_delta(s: Optional[float], m: Optional[float]) -> str:
    if s is None or m is None:
        return "   —  "
    d = m - s
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.3f}"


def print_table(methods: List[dict], n_concepts: int):
    w = 88
    print("\n" + "=" * w)
    print(f"Table 2 — Single (t2i) vs Multi  [concepts: ID@.68>{methods[0].get('id_thr', 0.5)}, n={n_concepts}]")
    print("=" * w)
    hdr = (f"{'Method':<12} {'TA (S→M Δ)':<22} {'IA (S→M Δ)':<22} "
           f"{'ID (S→M Δ)':<22} {'n':>6}")
    print(hdr)
    print("-" * w)
    s = methods[0]["single"]
    for m in methods:
        st, mt = s["ta"], m["multi"]["ta"]
        si, mi = s["ia"], m["multi"]["ia"]
        sid, mid = s["id"], m["multi"]["id"]
        ta_str = f"{st:.3f} → {mt:.3f}  ({fmt_delta(st, mt)})" if st and mt else "—"
        ia_str = f"{si:.3f} → {mi:.3f}  ({fmt_delta(si, mi)})" if si and mi else "—"
        id_str = f"{sid:.3f} → {mid:.3f}  ({fmt_delta(sid, mid)})" if sid and mid else "—"
        n_str = f"{m['multi'].get('n_kept', 0)}/{m['multi'].get('n_total', 0)}"
        print(f"{m['name']:<12} {ta_str:<22} {ia_str:<22} {id_str:<22} {n_str:>6}")
    print("=" * w)
    print("S=Single macro-avg (matched slugs: portrait/poker/cyberpunk)  "
          "M=Multi macro-avg (kept scenes)")
    print("Δ = M − S   (OrthA Table 2 convention)")


def main():
    ap = argparse.ArgumentParser(description="OrthA Table 2 S vs M summary")
    ap.add_argument("--single_metrics", default="outputs/eval_infer/metrics.json")
    ap.add_argument("--multi_root", default="outputs/multi_concept")
    ap.add_argument("--id_threshold", type=float, default=0.5)
    ap.add_argument("--output", default="outputs/ortha_table2.json")
    ap.add_argument("--matched_slugs", default="portrait,poker,cyberpunk")
    ap.add_argument("--methods", default="ortha,cascade,scaffold",
                    help="Comma-separated multi methods to include, each read from "
                         "{multi_root}/{method}/metrics.json. Missing ones are "
                         "skipped. 'ortha' is the same-backbone OrthA merge baseline.")
    args = ap.parse_args()

    slugs = tuple(s.strip() for s in args.matched_slugs.split(",") if s.strip())
    single = json.load(open(args.single_metrics))
    concepts = successful_concepts(single, args.id_threshold)
    s_macro = single_macro(single, concepts, slugs)

    multi_root = Path(args.multi_root)
    method_names = [m.strip() for m in args.methods.split(",") if m.strip()]
    methods = []
    for name in method_names:
        mpath = multi_root / name / "metrics.json"
        multi = load_multi_metrics(mpath)
        if not multi:
            continue
        mm = multi_macro(multi)
        methods.append({
            "name": name,
            "single": s_macro,
            "multi": mm,
            "id_thr": args.id_threshold,
            "delta": {
                "ta": (mm["ta"] - s_macro["ta"]) if mm["ta"] and s_macro["ta"] else None,
                "ia": (mm["ia"] - s_macro["ia"]) if mm["ia"] and s_macro["ia"] else None,
                "id": (mm["id"] - s_macro["id"]) if mm["id"] and s_macro["id"] else None,
            },
        })

    if not methods:
        raise SystemExit(
            f"No multi metrics under {multi_root}/{{{','.join(method_names)}}}/metrics.json")

    print_table(methods, len(concepts))

    out = {
        "n_concepts": len(concepts),
        "concepts": concepts,
        "matched_slugs": list(slugs),
        "multi_scene_map": MATCHED_SINGLE_SLUG,
        "single": s_macro,
        "methods": methods,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    csv_path = out_path.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "S_TA", "M_TA", "dTA", "S_IA", "M_IA", "dIA",
                    "S_ID", "M_ID", "dID", "n_kept", "n_total"])
        for m in methods:
            s, mm, d = m["single"], m["multi"], m["delta"]
            w.writerow([
                m["name"], s["ta"], mm["ta"], d["ta"],
                s["ia"], mm["ia"], d["ia"],
                s["id"], mm["id"], d["id"],
                mm.get("n_kept"), mm.get("n_total"),
            ])
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
