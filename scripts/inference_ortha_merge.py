#!/usr/bin/env python3
"""
OrthA (Orthogonal Adaptation, Po et al. CVPR 2024) multi-concept eval on
Qwen-Image-Edit — the REAL OrthA protocol, as the same-backbone baseline for
LILAC's cascade/scaffold.

Unlike cascade/scaffold (sequential, one LoRA per pass), OrthA:
  1. MERGES the per-concept LoRAs of a triple into the base transformer by
     summation of their weight deltas  ΔW_merged = Σ_i (α_i/r_i) · B_i · A_i.
     Because each concept's B_i was frozen to a disjoint orthonormal block of a
     shared basis (B_i^T B_j = 0, see train_lora_qwen_edit.py --ortha_orthogonal),
     the summed deltas interfere minimally.
  2. Generates every concept in a SINGLE forward pass from one prompt holding all
     trigger tokens: "<id_a> <cls_a>, <id_b> <cls_b>, and <id_c> <cls_c>, <scene>".

Outputs mirror the multi_concept layout so compute_metrics_multi.py
works unchanged:
  outputs/multi_concept/ortha/<triple_slug>/<scene>/seed<S>.png
  outputs/multi_concept/ortha/triples.json

Usage:
  python scripts/inference_ortha_merge.py \
      --suffix ortha --n_triples 8 --seeds "42 43 44 45"
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from safetensors.torch import load_file
from diffusers import QwenImageEditPipeline

from inference_lora import (
    _resolve_lora_weight_file,
    _convert_diffusers_to_peft_keys,
    _load_learned_embedding_if_present,
    generate_from_edit,
)
from inference_cascade import (
    SCENE_CONTEXTS,
    select_successful_concepts,
    form_triples,
    class_word,
    frobenius_score,
)
import random


# ─── LoRA delta extraction + merge ────────────────────────────────────────────

def _lora_scale(lora_path: str) -> float:
    cfg_path = os.path.join(lora_path, "training_config.json")
    r, alpha = 64, 64
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        r = cfg.get("rank", 64) or 64
        alpha = cfg.get("lora_alpha", r) or r
    return float(alpha) / float(r)


def _grouped_deltas(lora_path: str,
                    lora_scale_override: float = -1.0) -> Dict[str, torch.Tensor]:
    """Return {module_path: ΔW} for one concept LoRA, where
    ΔW = scale · B @ A  (shape [out_features, in_features]), computed in fp32.

    lora_scale_override: if >= 0, use this instead of alpha/r from training_config.
      The OrthA paper uses scale=1.0 at inference regardless of training alpha.
      The 'a2x' recipe has alpha/r=2.0 which is fine for single-concept
      inference but overdrive the model when 3 concepts are merged simultaneously.
    """
    sd = load_file(_resolve_lora_weight_file(lora_path))
    peft = _convert_diffusers_to_peft_keys(sd)
    scale = lora_scale_override if lora_scale_override >= 0 else _lora_scale(lora_path)

    mods: Dict[str, Dict[str, torch.Tensor]] = {}
    for k, v in peft.items():
        if k.endswith(".lora_A.weight"):
            mods.setdefault(k[: -len(".lora_A.weight")], {})["A"] = v
        elif k.endswith(".lora_B.weight"):
            mods.setdefault(k[: -len(".lora_B.weight")], {})["B"] = v

    deltas: Dict[str, torch.Tensor] = {}
    for mod, ab in mods.items():
        if "A" in ab and "B" in ab:
            A = ab["A"].float()  # [r, in]
            B = ab["B"].float()  # [out, r]
            deltas[mod] = scale * (B @ A)  # [out, in]
    return deltas


def snapshot_base_weights(transformer, module_paths: List[str]) -> Dict[str, torch.Tensor]:
    """Clone (to CPU, fp32) the base weights that the merge will overwrite, so we
    can restore them exactly before merging the next triple."""
    snap = {}
    for mp in module_paths:
        try:
            lin = transformer.get_submodule(mp)
        except AttributeError:
            continue
        snap[mp] = lin.weight.detach().to("cpu", torch.float32).clone()
    return snap


def restore_base_weights(transformer, snapshot: Dict[str, torch.Tensor]) -> None:
    for mp, w in snapshot.items():
        lin = transformer.get_submodule(mp)
        with torch.no_grad():
            lin.weight.data.copy_(w.to(lin.weight.device, lin.weight.dtype))


def merge_triple(transformer, lora_paths: List[str],
                 lora_scale_override: float = -1.0) -> int:
    """Add Σ_i ΔW_i into the matching base linear weights (in place). Assumes the
    base weights are already at their pristine (restored) values. Returns the
    number of (module, concept) deltas applied."""
    applied = 0
    for lp in lora_paths:
        for mp, dW in _grouped_deltas(lp, lora_scale_override).items():
            try:
                lin = transformer.get_submodule(mp)
            except AttributeError:
                continue
            with torch.no_grad():
                lin.weight.data.add_(dW.to(lin.weight.device, lin.weight.dtype))
            applied += 1
    return applied


# ─── Prompt ───────────────────────────────────────────────────────────────────

_POSITIONS = ["on the far left", "in the center", "on the far right",
              "in the background left", "in the background right"]


def ortha_prompt(triggers: List[str], classes: List[str], scene_ctx: str) -> str:
    """Single-pass merged-LoRA prompt with explicit spatial anchors per concept.

    Spatial anchors (far left / center / far right) help the model's cross-attention
    maps associate each trigger token with a distinct spatial region, reducing the
    identity bleed that occurs when all triggers compete globally for the same pixels.
    """
    parts = []
    for i, (t, c) in enumerate(zip(triggers, classes)):
        pos = _POSITIONS[i] if i < len(_POSITIONS) else "in the scene"
        parts.append(f"{t} {c} {pos}")
    people = ", ".join(parts[:-1]) + f", and {parts[-1]}" if len(parts) > 1 else parts[0]
    return f"a photo of {people}, {scene_ctx}, each person clearly distinct"


# ─── Concept pool (concepts that have OrthA weights) ──────────────────────────

def detect_ortha_concepts(weights_root: str, suffix: str, registry_map: dict) -> List[str]:
    out: List[str] = []
    root = Path(weights_root)
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        tail = f"_lora_{suffix}"
        if not name.endswith(tail):
            continue
        concept = name[: -len(tail)]
        if concept in registry_map and (d / "pytorch_lora_weights.safetensors").exists():
            out.append(concept)
    return out


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="OrthA merge multi-concept eval (Qwen, single-pass)")
    ap.add_argument("--base_model", default="Qwen/Qwen-Image-Edit")
    ap.add_argument("--registry", default="concept_registry.json")
    ap.add_argument("--metrics", default="outputs/eval_infer/metrics.json")
    ap.add_argument("--weights_root", default="outputs/lora_weights")
    ap.add_argument("--suffix", default="ortha")
    ap.add_argument("--out_root", default="outputs/multi_concept")
    ap.add_argument("--id_threshold", type=float, default=0.5)
    ap.add_argument("--n_triples", type=int, default=8)
    ap.add_argument("--triples", default="",
                    help="Explicit triples 'c1,c2,c3;c4,c5,c6'. Overrides --n_triples.")
    ap.add_argument("--concepts", default="",
                    help="Restrict the triple pool to these space-separated concepts. "
                         "Default: auto-detect all concepts with OrthA weights.")
    ap.add_argument("--require_success", action="store_true",
                    help="Also require pool concepts to pass single-concept t2i ID "
                         "(> --id_threshold from --metrics). Off by default.")
    ap.add_argument("--seeds", default="42 43 44 45",
                    help="Seeds for best-of-N selection at eval time.")
    ap.add_argument("--scenes", default="plain poker cyberpunk")
    ap.add_argument("--num_steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--lora_scale", type=float, default=1.0,
                    help="Per-concept LoRA scale at merge time. Default 1.0 matches the "
                         "OrthA paper (alpha/r is used during training but 1.0 at inference). "
                         "The 'a2x' recipe has alpha/r=2.0 which overdrives "
                         "the model when 3 concepts are merged simultaneously; 1.0 avoids "
                         "the high-frequency grid artifact. Use -1 to use alpha/r from "
                         "each concept's training_config.json.")
    ap.add_argument("--width", type=int, default=1536)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--rng_seed", type=int, default=0)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split()]
    scenes = args.scenes.split()
    out_root = Path(args.out_root) / "ortha"

    registry = json.load(open(args.registry))
    registry_map = {e["concept"]: e for e in registry if e.get("concept")}

    # ── Build the concept pool ──
    if args.concepts.strip():
        pool = [c for c in args.concepts.split() if c in registry_map]
    else:
        pool = detect_ortha_concepts(args.weights_root, args.suffix, registry_map)
    if args.require_success and Path(args.metrics).exists():
        metrics = json.load(open(args.metrics))
        ok = set(select_successful_concepts(metrics, registry_map, args.id_threshold))
        pool = [c for c in pool if c in ok]
    pool = sorted(pool)
    print(f"OrthA concept pool ({len(pool)}): {pool or '(none)'}")

    # ── Triples (same logic/seed as cascade => identical triples) ──
    rng = random.Random(args.rng_seed)
    if args.triples.strip():
        triples = []
        for spec in args.triples.split(";"):
            cs = [c.strip() for c in spec.split(",") if c.strip()]
            if len(cs) < 2:
                continue
            for c in cs:
                if c not in registry_map:
                    raise SystemExit(f"--triples concept '{c}' not in registry")
            triples.append(cs)
        if not triples:
            raise SystemExit("--triples given but no valid triple parsed.")
    else:
        if len(pool) < 3:
            raise SystemExit(f"Need >= 3 OrthA concepts to form triples, got {len(pool)}.")
        triples = form_triples(pool, args.n_triples, rng)

    # Order each triple by Frobenius norm (same convention as cascade -> matching slugs).
    fro_cache: Dict[str, float] = {}
    def fro(concept):
        if concept not in fro_cache:
            fro_cache[concept] = frobenius_score(
                f"{args.weights_root}/{concept}_lora_{args.suffix}")
        return fro_cache[concept]

    ordered_triples = []
    for tri in triples:
        ts = sorted(tri, key=fro, reverse=True)
        ordered_triples.append(ts)
        print(f"  triple {ts}  (||ΔW||F={[round(fro(c), 1) for c in ts]})")

    out_root.mkdir(parents=True, exist_ok=True)
    if args.shard_id == 0:
        manifest = {
            "triples": ["__".join(t) for t in ordered_triples],
            "order": {"__".join(t): t for t in ordered_triples},
            "scenes": scenes,
            "seeds": seeds,
            "width": args.width,
            "height": args.height,
            "method": "ortha",
        }
        with open(out_root / "triples.json", "w") as f:
            json.dump(manifest, f, indent=2)

    # ── Load base model once ──
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — OrthA generation would run on CPU.")
    print(f"Loading base model: {args.base_model}")
    pipe = QwenImageEditPipeline.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    dev = next(pipe.transformer.parameters()).device
    print(f"  pipe.device={pipe.device}  transformer params on {dev}")
    if dev.type != "cuda":
        raise SystemExit(f"Transformer is on {dev}, expected cuda. Aborting.")

    # ── Snapshot the base weights that any concept's merge will touch ──
    # All OrthA concepts share the same target modules, so the union == one
    # concept's module set. Build it from the first available LoRA in the pool.
    probe = None
    for c in pool or [t for tri in ordered_triples for t in tri]:
        lp = f"{args.weights_root}/{c}_lora_{args.suffix}"
        if os.path.exists(os.path.join(lp, "pytorch_lora_weights.safetensors")):
            probe = lp
            break
    if probe is None:
        raise SystemExit("No OrthA LoRA weights found to probe target modules.")
    module_paths = list(_grouped_deltas(probe).keys())
    print(f"  merge touches {len(module_paths)} transformer linear modules")
    base_snapshot = snapshot_base_weights(pipe.transformer, module_paths)

    # ── Triple-level jobs (merge once per triple, then all scenes/seeds) ──
    if args.num_shards > 1:
        my_triples = [t for i, t in enumerate(ordered_triples)
                      if i % args.num_shards == args.shard_id]
        print(f"[shard {args.shard_id}/{args.num_shards}] handling "
              f"{len(my_triples)}/{len(ordered_triples)} triples")
    else:
        my_triples = ordered_triples

    for tri in my_triples:
        tri_slug = "__".join(tri)
        lora_paths = [f"{args.weights_root}/{c}_lora_{args.suffix}" for c in tri]
        missing = [c for c, lp in zip(tri, lora_paths)
                   if not os.path.exists(os.path.join(lp, "pytorch_lora_weights.safetensors"))]
        if missing:
            print(f"  SKIP {tri_slug} — missing OrthA weights for {missing}")
            continue

        # Decide whether any output for this triple is still needed.
        triggers = [registry_map[c]["trigger"] for c in tri]
        classes = [class_word(registry_map[c]) for c in tri]
        jobs = []
        for scene in scenes:
            for seed in seeds:
                fp = out_root / tri_slug / scene / f"seed{seed}.png"
                if fp.exists() and not args.force:
                    continue
                jobs.append((scene, seed, fp))
        if not jobs:
            print(f"  skip {tri_slug} (all outputs exist)")
            continue

        # Merge this triple's deltas into the base (restore first for determinism).
        restore_base_weights(pipe.transformer, base_snapshot)
        n = merge_triple(pipe.transformer, lora_paths,
                         lora_scale_override=args.lora_scale)
        for lp in lora_paths:
            _load_learned_embedding_if_present(pipe, lp)
        print(f"[{tri_slug}] merged {n} module-deltas "
              f"(scale={args.lora_scale}); generating {len(jobs)} images")

        source = Image.new("RGB", (args.width, args.height), (128, 128, 128))
        for scene, seed, fp in jobs:
            scene_ctx = SCENE_CONTEXTS.get(scene, scene)
            prompt = ortha_prompt(triggers, classes, scene_ctx)
            fp.parent.mkdir(parents=True, exist_ok=True)
            print(f"    scene={scene} seed={seed} :: {prompt}")
            img = generate_from_edit(
                pipe, source, prompt,
                seed=seed, num_steps=args.num_steps, cfg_scale=args.cfg,
            )
            img.save(fp)
            print(f"    saved {fp}")

    # Leave the base model pristine on exit.
    restore_base_weights(pipe.transformer, base_snapshot)

    print("\nDone. Next:")
    print(f"  python scripts/compute_metrics_multi.py --multi_dir {out_root} "
          f"--rank_metric id_min --min_per_concept_id 1.0 --min_ia 0.55 --min_ta 0.55")


if __name__ == "__main__":
    main()
