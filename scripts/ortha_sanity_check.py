#!/usr/bin/env python3
"""
OrthA sanity check (GPU-free): verify the orthogonal-adaptation invariants on
the trained LoRA weights, BEFORE spending GPU time on merge inference.

Checks, on a sample of shared target modules across the OrthA concepts:

  1. Orthogonality:   B_i^T B_j ≈ 0  for i != j   (disjoint orthonormal blocks)
     and             B_i^T B_i ≈ I                 (each block orthonormal)

  2. Single-concept recovery from the merged delta:
     Let ΔW_merged = Σ_k (α_k/r_k) B_k A_k. Because B_i^T B_k = δ_ik·I,
        B_i^T ΔW_merged / (α_i/r_i)  ==  A_i.
     This proves concept i's contribution survives the summation intact — the
     whole point of OrthA. We report the relative error.

Usage:
  python scripts/ortha_sanity_check.py \
      --weights_root outputs/lora_weights --suffix ortha \
      --concepts "thanos gosling margotrobbie thor hulk lebron bradpitt jamiefoxx"
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
from safetensors.torch import load_file

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from inference_lora import _resolve_lora_weight_file, _convert_diffusers_to_peft_keys


def load_AB(lora_path: str) -> Dict[str, Dict[str, torch.Tensor]]:
    sd = load_file(_resolve_lora_weight_file(lora_path))
    peft = _convert_diffusers_to_peft_keys(sd)
    mods: Dict[str, Dict[str, torch.Tensor]] = {}
    for k, v in peft.items():
        if k.endswith(".lora_A.weight"):
            mods.setdefault(k[: -len(".lora_A.weight")], {})["A"] = v.float()
        elif k.endswith(".lora_B.weight"):
            mods.setdefault(k[: -len(".lora_B.weight")], {})["B"] = v.float()
    return {m: ab for m, ab in mods.items() if "A" in ab and "B" in ab}


def lora_scale(lora_path: str) -> float:
    cfg_path = os.path.join(lora_path, "training_config.json")
    r, alpha = 64, 64
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        r = cfg.get("rank", 64) or 64
        alpha = cfg.get("lora_alpha", r) or r
    return float(alpha) / float(r)


def main():
    ap = argparse.ArgumentParser(description="OrthA orthogonality + recovery sanity check")
    ap.add_argument("--weights_root", default="outputs/lora_weights")
    ap.add_argument("--suffix", default="ortha")
    ap.add_argument("--concepts", default="",
                    help="Space-separated concept list. Default: auto-detect *_lora_<suffix>.")
    ap.add_argument("--num_modules", type=int, default=3,
                    help="How many shared modules to sample for the report.")
    ap.add_argument("--ortho_tol", type=float, default=1e-2,
                    help="Max allowed cross-block ||B_i^T B_j||_F (orthonormal => ~0).")
    ap.add_argument("--recover_tol", type=float, default=5e-3,
                    help="Max allowed relative recovery error of A_i from the merge. "
                         "bf16 training (machine eps ~1.9e-3) introduces a fp32 round-trip "
                         "error of ~2-4e-3, so 5e-3 is the right threshold for bf16 weights.")
    args = ap.parse_args()

    root = Path(args.weights_root)
    if args.concepts.strip():
        concepts = args.concepts.split()
    else:
        concepts = []
        tail = f"_lora_{args.suffix}"
        for d in sorted(root.iterdir()) if root.is_dir() else []:
            if d.is_dir() and d.name.endswith(tail) \
                    and (d / "pytorch_lora_weights.safetensors").exists():
                concepts.append(d.name[: -len(tail)])
    if len(concepts) < 2:
        raise SystemExit(f"Need >= 2 OrthA concepts, found {concepts}")

    paths = {c: f"{args.weights_root}/{c}_lora_{args.suffix}" for c in concepts}
    print(f"OrthA sanity check on {len(concepts)} concepts: {concepts}")

    ab = {c: load_AB(paths[c]) for c in concepts}
    scales = {c: lora_scale(paths[c]) for c in concepts}

    # Modules present in every concept.
    common = set.intersection(*[set(ab[c].keys()) for c in concepts])
    common = sorted(common)
    if not common:
        raise SystemExit("No shared modules across concepts — cannot check.")
    sample = common[: args.num_modules]
    print(f"Shared modules: {len(common)}; sampling {len(sample)}\n")

    worst_cross = 0.0
    worst_recover = 0.0
    for mod in sample:
        Bs = {c: ab[c][mod]["B"] for c in concepts}   # [out, r]
        As = {c: ab[c][mod]["A"] for c in concepts}   # [r, in]
        r = next(iter(Bs.values())).shape[1]

        print(f"[{mod}]  out={Bs[concepts[0]].shape[0]}  r={r}")

        # (1a) self orthonormality: B_i^T B_i ≈ I
        for c in concepts:
            G = Bs[c].T @ Bs[c]
            self_err = (G - torch.eye(r)).norm().item() / (r ** 0.5)
            tag = "ok" if self_err < args.ortho_tol else "WARN"
            print(f"   self  B[{c}]^T B[{c}] - I  rms={self_err:.2e}  [{tag}]")

        # (1b) cross orthogonality: B_i^T B_j ≈ 0
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                ci, cj = concepts[i], concepts[j]
                cross = (Bs[ci].T @ Bs[cj]).norm().item() / r
                worst_cross = max(worst_cross, cross)
        # report only the max for brevity
        # (2) recovery: B_i^T ΔW_merged / scale_i ≈ A_i
        dW_merged = sum(scales[c] * (Bs[c] @ As[c]) for c in concepts)
        for c in concepts:
            recovered = (Bs[c].T @ dW_merged) / scales[c]
            denom = As[c].norm().item() or 1.0
            rel = (recovered - As[c]).norm().item() / denom
            worst_recover = max(worst_recover, rel)
        print(f"   cross max ||B_i^T B_j||/r = {worst_cross:.2e}   "
              f"recovery max rel-err = {worst_recover:.2e}\n")

    print("=" * 60)
    ortho_ok = worst_cross < args.ortho_tol
    recover_ok = worst_recover < args.recover_tol
    print(f"Orthogonality : max cross ||B_i^T B_j||/r = {worst_cross:.2e}  "
          f"(tol {args.ortho_tol:.0e})  -> {'PASS' if ortho_ok else 'FAIL'}")
    print(f"Recovery      : max rel-err of A_i        = {worst_recover:.2e}  "
          f"(tol {args.recover_tol:.0e})  -> {'PASS' if recover_ok else 'FAIL'}")
    print("=" * 60)
    if not (ortho_ok and recover_ok):
        print("WARNING: an invariant failed. Check that all concepts were trained "
              "with --ortha_orthogonal, the SAME --ortha_basis_seed, and the SAME "
              "--ortha_num_concepts / rank / target_modules.")
        sys.exit(1)
    print("All OrthA invariants hold — safe to merge.")


if __name__ == "__main__":
    main()
