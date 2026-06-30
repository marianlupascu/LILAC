#!/usr/bin/env python3
"""
Multi-concept eval via CASCADED LAYER inference on Qwen-Image-Edit.

Implements the layer-wise low-rank adaptation method with frozen conditioning,
realized on Qwen-Image-Edit:

  - Pass 1 (anchor): blank 3:1 canvas, only the anchor concept's
    LoRA active -> generates subject 1.
  - Pass k: the frozen output of pass k-1 is used as the
    source_image conditioning, only LoRA_k active -> adds subject k while
    preserving the existing composite.
  - Exactly one LoRA active per pass = per-layer concept binding.
  - Anchor = concept with the largest LoRA delta Frobenius norm (default).

Evaluated concepts are the *successful* ones: persons with single-concept
t2i Identity Alignment (ArcFace detection rate @0.68) > --id_threshold,
read from outputs/eval_infer/metrics.json. Triples are formed at random
and rendered at 3:1 landscape, following the multi-concept eval protocol.

Outputs:
  outputs/multi_concept/<triplet_slug>/<scene>/seed<S>.png        (final)
  outputs/multi_concept/<triplet_slug>/<scene>/seed<S>_pass<n>.png (layers)
  outputs/multi_concept/triples.json                              (manifest)

Usage:
  python scripts/inference_cascade.py \
      --registry concept_registry.json \
      --metrics outputs/eval_infer/metrics.json \
      --n_triples 5 --seeds "42 43"
"""

import argparse
import itertools
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image

# Reuse the (tested) single-concept loaders/generator.
from inference_lora import (
    _detect_format,
    _load_peft_legacy,
    _load_learned_embedding_if_present,
    _resolve_lora_weight_file,
    _convert_diffusers_to_peft_keys,
    generate_from_edit,
)
from diffusers import QwenImageEditPipeline
from safetensors.torch import load_file


# ─── Scene templates ──────────────────────────────────────────────────────────
# scene_ctx is appended to every per-pass prompt for that scene.
SCENE_CONTEXTS = {
    "plain":     "group portrait, neutral background, studio lighting",
    "poker":     "playing poker together, cinematic lighting, dramatic",
    "cyberpunk": "in the style of Cyberpunk 2077, 4K, ultra-realistic, neon",
}

# Positional hints for the additive cascade, indexed by layer order.
POSITIONS = ["on the left", "on the right", "in the center",
             "in the foreground", "in the background"]

# Left-to-right positions for the scaffold mode (concept i -> person i).
SCAFFOLD_POSITIONS = ["on the left", "in the center", "on the right",
                      "in the back left", "in the back right"]


# ─── Anchor selection: LoRA delta Frobenius norm ──────────────────────────────

def frobenius_score(lora_path: str) -> float:
    """Proxy of ||ΔW||_F = sum over modules of (alpha/r)*||A||_F*||B||_F."""
    try:
        weight_file = _resolve_lora_weight_file(lora_path)
    except FileNotFoundError:
        return 0.0
    sd = load_file(weight_file)

    cfg = {}
    cfg_path = os.path.join(lora_path, "training_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
    r = cfg.get("rank", 64) or 64
    alpha = cfg.get("lora_alpha", r) or r
    scale = float(alpha) / float(r)

    peft = _convert_diffusers_to_peft_keys(sd)
    mods: Dict[str, Dict[str, torch.Tensor]] = {}
    for k, v in peft.items():
        if k.endswith(".lora_A.weight"):
            mods.setdefault(k[: -len(".lora_A.weight")], {})["A"] = v
        elif k.endswith(".lora_B.weight"):
            mods.setdefault(k[: -len(".lora_B.weight")], {})["B"] = v

    total_sq = 0.0
    for d in mods.values():
        if "A" in d and "B" in d:
            na = d["A"].float().norm().item()
            nb = d["B"].float().norm().item()
            total_sq += (scale * na * nb) ** 2
    return total_sq ** 0.5


# ─── Per-pass LoRA binding (swap adapter, keep base loaded) ────────────────────

def _delete_existing_adapter(transformer) -> None:
    for fn, arg in (("delete_adapters", ["default"]),
                    ("delete_adapter", "default")):
        method = getattr(transformer, fn, None)
        if method is None:
            continue
        try:
            method(arg)
            return
        except Exception:
            continue


def clear_lora(pipe) -> None:
    """Remove any active adapter so the next pass runs on the pure base model
    (used for the scaffold pass in scaffold mode)."""
    _delete_existing_adapter(pipe.transformer)


def apply_concept_lora(pipe, lora_path: str, sd_cache: Dict[str, tuple],
                       device: str = "cuda") -> None:
    """Bind a single concept's LoRA (+TI) to the pipeline, replacing any prior
    adapter. Keeps the base model resident on GPU between passes."""
    _delete_existing_adapter(pipe.transformer)

    if lora_path in sd_cache:
        _, state_dict = sd_cache[lora_path]
    else:
        fmt, state_dict = _detect_format(lora_path)
        sd_cache[lora_path] = (fmt, state_dict)

    _load_peft_legacy(pipe, state_dict, lora_path)
    # Use an explicit device (NOT pipe.device, which may report cpu and would
    # silently move the whole transformer off-GPU after every adapter swap).
    pipe.transformer.to(device)
    _load_learned_embedding_if_present(pipe, lora_path)


# ─── Concept selection + triple formation ─────────────────────────────────────

def select_successful_concepts(metrics: dict, registry_map: dict,
                               id_threshold: float) -> List[str]:
    out = []
    for concept, cr in metrics.items():
        if concept not in registry_map:
            continue
        t2i = cr.get("t2i", {}) or {}
        id_score = (t2i.get("avg", {}) or {}).get("id")
        if id_score is not None and id_score > id_threshold:
            out.append(concept)
    return sorted(out)


def form_triples(concepts: List[str], n_triples: int, rng: random.Random) -> List[List[str]]:
    if len(concepts) < 3:
        return []
    all_combos = list(itertools.combinations(concepts, 3))
    rng.shuffle(all_combos)
    n = min(n_triples, len(all_combos))
    return [list(c) for c in all_combos[:n]]


# ─── Prompt construction ──────────────────────────────────────────────────────

def class_word(entry: dict) -> str:
    anchors = entry.get("attribute_anchors") or []
    return anchors[0] if anchors else entry.get("class_noun", "a person")


def gender_word(entry: dict) -> str:
    """Generic placeholder phrase for the scaffold, by gender."""
    g = (entry.get("gender") or "").lower()
    if g in ("man", "woman"):
        return f"a {g}"
    cn = entry.get("class_noun")
    if cn:
        return f"a {cn}"
    return class_word(entry)


def pass_prompt(layer_idx: int, trigger: str, cls: str, scene_ctx: str) -> str:
    """Additive cascade per-pass prompt."""
    pos = POSITIONS[layer_idx] if layer_idx < len(POSITIONS) else "in the scene"
    if layer_idx == 0:
        return f"{trigger}, {cls}, {scene_ctx}, positioned {pos}"
    return (f"add {trigger}, {cls}, {pos}, {scene_ctx}, "
            f"keep the existing people unchanged")


def scaffold_prompt(genders: List[str], scene_ctx: str) -> str:
    """Pass-0 scaffold: N generic people of the right gender, in scene style."""
    people = ", ".join(
        f"{g} {SCAFFOLD_POSITIONS[i] if i < len(SCAFFOLD_POSITIONS) else 'in the scene'}"
        for i, g in enumerate(genders))
    return (f"a photo of exactly {len(genders)} people: {people}, "
            f"{scene_ctx}, standing side by side, group composition")


def replace_prompt(layer_idx: int, trigger: str, cls: str, scene_ctx: str) -> str:
    """Scaffold replace pass: swap one placeholder for a concept identity."""
    pos = SCAFFOLD_POSITIONS[layer_idx] if layer_idx < len(SCAFFOLD_POSITIONS) else "in the scene"
    return (f"replace the person {pos} with {trigger}, {cls}, "
            f"keep the other people and the background unchanged, {scene_ctx}")


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _first_ref_image(entry: dict, datasets_dir: str) -> Optional[str]:
    ds = Path(entry.get("dataset_dir", ""))
    if not ds.is_dir():
        ds = Path(datasets_dir) / entry.get("dataset", "")
    if not ds.is_dir():
        return None
    for p in sorted(ds.iterdir()):
        if p.suffix.lower() in IMAGE_EXTS:
            return str(p)
    return None


def build_ref_strip_canvas(
    concepts: List[str],
    registry_map: dict,
    datasets_dir: str,
    width: int,
    height: int,
) -> Image.Image:
    """3:1 canvas with one training ref crop per concept (left/center/right)."""
    canvas = Image.new("RGB", (width, height), (128, 128, 128))
    n = max(len(concepts), 1)
    slot_w = width // n
    for i, concept in enumerate(concepts):
        entry = registry_map.get(concept, {})
        ref = _first_ref_image(entry, datasets_dir)
        if not ref:
            continue
        img = Image.open(ref).convert("RGB")
        img = img.resize((slot_w, height), Image.Resampling.LANCZOS)
        canvas.paste(img, (i * slot_w, 0))
    return canvas


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Cascaded multi-concept eval (Qwen)")
    ap.add_argument("--base_model", default="Qwen/Qwen-Image-Edit")
    ap.add_argument("--registry", default="concept_registry.json")
    ap.add_argument("--metrics", default="outputs/eval_infer/metrics.json")
    ap.add_argument("--weights_root", default="outputs/lora_weights")
    ap.add_argument("--suffix", default="ti_a2x")
    ap.add_argument("--out_root", default="outputs/multi_concept")
    ap.add_argument("--id_threshold", type=float, default=0.5)
    ap.add_argument("--n_triples", type=int, default=5)
    ap.add_argument("--triples", default="",
                    help="Explicit triples instead of random selection. "
                         "Format: 'c1,c2,c3;c4,c5,c6'. Overrides --n_triples. "
                         "Example: 'thanos,gosling,margotrobbie'.")
    ap.add_argument("--seeds", default="42 43 44 45",
                    help="Multiple seeds for best-of-N selection at eval time.")
    ap.add_argument("--scenes", default="plain poker cyberpunk")
    ap.add_argument("--num_steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--width", type=int, default=1536)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--rng_seed", type=int, default=0)
    ap.add_argument("--shard_id", type=int, default=0,
                    help="This worker's index (0..num_shards-1) for multi-GPU.")
    ap.add_argument("--num_shards", type=int, default=1,
                    help="Total workers; each handles triples where idx%%num_shards==shard_id.")
    ap.add_argument("--mode", default="cascade", choices=["cascade", "scaffold"],
                    help="cascade=additive (add subjects one by one); "
                         "scaffold=generate generic people first, then replace each.")
    ap.add_argument("--scaffold_init", default="gray", choices=["gray", "ref_strip"],
                    help="scaffold pass-0 init: gray canvas or 3 training ref crops on 3:1 layout.")
    ap.add_argument("--datasets_dir", default="Datasets",
                    help="Dataset root for ref_strip scaffold init.")
    ap.add_argument("--save_layers", action="store_true", default=True,
                    help="Save intermediate cascade layers (default on).")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split()]
    scenes = args.scenes.split()
    # Separate output tree per mode so cascade vs scaffold can be compared.
    out_root = Path(args.out_root) / args.mode

    # ── Load registry + metrics ──
    registry = json.load(open(args.registry))
    registry_map = {e["concept"]: e for e in registry if e.get("concept")}

    if not Path(args.metrics).exists():
        raise SystemExit(
            f"Metrics not found: {args.metrics}. Run compute_metrics.py first "
            f"so the t2i ID@0.68 scores exist for concept selection.")
    metrics = json.load(open(args.metrics))

    succeeded = select_successful_concepts(metrics, registry_map, args.id_threshold)
    print(f"Successful concepts (t2i ID@.68 > {args.id_threshold}): "
          f"{succeeded or '(none)'}")

    rng = random.Random(args.rng_seed)
    if args.triples.strip():
        # Explicit triples (e.g. the paper's <THANOS> & <RYAN> & <MARGOT>).
        triples = []
        for spec in args.triples.split(";"):
            cs = [c.strip() for c in spec.split(",") if c.strip()]
            if len(cs) < 2:
                continue
            for c in cs:
                if c not in registry_map:
                    raise SystemExit(f"--triples concept '{c}' not in registry")
                if c not in succeeded:
                    print(f"  WARN: '{c}' is not in the successful set "
                          f"(t2i ID<= {args.id_threshold}) but included via --triples")
            triples.append(cs)
        if not triples:
            raise SystemExit("--triples given but no valid triple parsed.")
        print(f"Using {len(triples)} explicit triple(s) from --triples")
    else:
        if len(succeeded) < 3:
            raise SystemExit(
                f"Need >= 3 successful concepts to form triples, got {len(succeeded)}.")
        triples = form_triples(succeeded, args.n_triples, rng)

    # Order each triple by Frobenius norm (anchor first).
    fro_cache: Dict[str, float] = {}
    def fro(concept):
        if concept not in fro_cache:
            lp = f"{args.weights_root}/{concept}_lora_{args.suffix}"
            fro_cache[concept] = frobenius_score(lp)
        return fro_cache[concept]

    ordered_triples = []
    for tri in triples:
        tri_sorted = sorted(tri, key=fro, reverse=True)
        ordered_triples.append(tri_sorted)
        print(f"  triple {tri_sorted}  (anchor={tri_sorted[0]}, "
              f"||ΔW||F={[round(fro(c), 1) for c in tri_sorted]})")

    # ── Manifest (used by metrics + html) — only shard 0 writes it ──
    # All shards compute the same deterministic ordered_triples (fixed rng_seed),
    # so the manifest is identical; writing from one shard avoids a race.
    out_root.mkdir(parents=True, exist_ok=True)
    if args.shard_id == 0:
        manifest = {
            "triples": ["__".join(t) for t in ordered_triples],
            "order": {"__".join(t): t for t in ordered_triples},
            "scenes": scenes,
            "seeds": seeds,
            "width": args.width,
            "height": args.height,
        }
        with open(out_root / "triples.json", "w") as f:
            json.dump(manifest, f, indent=2)

    # ── Shard at (triple, seed) granularity so even a single explicit triple
    #    with many seeds spreads across all GPUs. ──
    all_jobs = [(tri, seed) for tri in ordered_triples for seed in seeds]
    if args.num_shards > 1:
        my_jobs = [j for i, j in enumerate(all_jobs)
                   if i % args.num_shards == args.shard_id]
        print(f"[shard {args.shard_id}/{args.num_shards}] handling "
              f"{len(my_jobs)}/{len(all_jobs)} (triple,seed) jobs")
    else:
        my_jobs = all_jobs

    # ── Load base model once ──
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available to this process — generation would run on CPU "
            "(unusably slow). Check torch/CUDA install and CUDA_VISIBLE_DEVICES.")
    print(f"Loading base model: {args.base_model}")
    pipe = QwenImageEditPipeline.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    # Sanity: confirm the model actually landed on GPU.
    dev = next(pipe.transformer.parameters()).device
    print(f"  pipe.device={pipe.device}  transformer params on {dev}")
    if dev.type != "cuda":
        raise SystemExit(f"Transformer is on {dev}, expected cuda. Aborting.")

    sd_cache: Dict[str, tuple] = {}

    # Per-triple info cache (triggers / classes / lora paths).
    tri_info: Dict[str, dict] = {}
    def info_for(tri):
        slug = "__".join(tri)
        if slug not in tri_info:
            tri_info[slug] = {
                "triggers": [registry_map[c]["trigger"] for c in tri],
                "classes": [class_word(registry_map[c]) for c in tri],
                "lora_paths": [f"{args.weights_root}/{c}_lora_{args.suffix}" for c in tri],
            }
        return tri_info[slug]

    # ── Generation (one (triple, seed) job at a time) ──
    for tri, seed in my_jobs:
        tri_slug = "__".join(tri)
        info = info_for(tri)
        triggers = info["triggers"]
        classes = info["classes"]
        lora_paths = info["lora_paths"]

        # Skip triple if any LoRA missing.
        missing = [c for c, lp in zip(tri, lora_paths)
                   if not os.path.exists(os.path.join(lp, "pytorch_lora_weights.safetensors"))]
        if missing:
            print(f"  SKIP {tri_slug} seed{seed} — missing weights for {missing}")
            continue

        for scene in scenes:
            scene_ctx = SCENE_CONTEXTS.get(scene, scene)
            out_dir = out_root / tri_slug / scene
            out_dir.mkdir(parents=True, exist_ok=True)

            final_path = out_dir / f"seed{seed}.png"
            if final_path.exists() and not args.force:
                print(f"  skip existing {final_path}")
                continue

            if True:
                print(f"[{tri_slug}] scene={scene} seed={seed} mode={args.mode}")
                source = Image.new("RGB", (args.width, args.height), (128, 128, 128))
                composite = None

                if args.mode == "scaffold":
                    if args.scaffold_init == "ref_strip":
                        composite = build_ref_strip_canvas(
                            tri, registry_map, args.datasets_dir,
                            args.width, args.height,
                        )
                        print(f"    pass0 [ref_strip] training ref crops")
                        if args.save_layers:
                            composite.save(out_dir / f"seed{seed}_pass0.png")
                    else:
                        clear_lora(pipe)
                        genders = [gender_word(registry_map[c]) for c in tri]
                        s_prompt = scaffold_prompt(genders, scene_ctx)
                        print(f"    pass0 [scaffold] {s_prompt}")
                        composite = generate_from_edit(
                            pipe, source, s_prompt,
                            seed=seed, num_steps=args.num_steps, cfg_scale=args.cfg,
                        )
                        if args.save_layers:
                            composite.save(out_dir / f"seed{seed}_pass0.png")

                    # Replace passes: swap each placeholder for its concept.
                    for layer_idx, concept in enumerate(tri):
                        apply_concept_lora(pipe, lora_paths[layer_idx], sd_cache,
                                           device="cuda")
                        r_prompt = replace_prompt(layer_idx, triggers[layer_idx],
                                                  classes[layer_idx], scene_ctx)
                        print(f"    pass{layer_idx+1} [{concept}] {r_prompt}")
                        composite = generate_from_edit(
                            pipe, composite, r_prompt,
                            seed=seed, num_steps=args.num_steps, cfg_scale=args.cfg,
                        )
                        if args.save_layers:
                            composite.save(out_dir / f"seed{seed}_pass{layer_idx+1}.png")
                else:
                    # Additive cascade: add subjects one at a time.
                    for layer_idx, concept in enumerate(tri):
                        apply_concept_lora(pipe, lora_paths[layer_idx], sd_cache,
                                           device="cuda")
                        prompt = pass_prompt(layer_idx, triggers[layer_idx],
                                             classes[layer_idx], scene_ctx)
                        src = source if layer_idx == 0 else composite
                        print(f"    pass{layer_idx+1} [{concept}] {prompt}")
                        composite = generate_from_edit(
                            pipe, src, prompt,
                            seed=seed, num_steps=args.num_steps, cfg_scale=args.cfg,
                        )
                        if args.save_layers:
                            composite.save(out_dir / f"seed{seed}_pass{layer_idx+1}.png")

                composite.save(final_path)
                print(f"    saved {final_path}")

    print("\nDone. Next:")
    print(f"  python scripts/compute_metrics_multi.py --multi_dir {out_root}")


if __name__ == "__main__":
    main()
