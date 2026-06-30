"""
Inference script for testing trained LoRA weights on Qwen-Image-Edit.

Handles both:
  - diffusers format (saved via Pipeline.save_lora_weights) → native loading
  - legacy peft format (saved via raw save_file of peft state dict) → manual loading

Usage:
    # Single concept inference (gray source → tests pure text-based identity)
    python scripts/inference_lora.py \
        --lora_path outputs/lora_weights/trump_lora \
        --prompt "a photo of ohwx_trump standing in front of the Eiffel Tower" \
        --output_dir outputs/generated/trump_test \
        --num_images 4

    # With source image (edit mode)
    python scripts/inference_lora.py \
        --lora_path outputs/lora_weights/pope_lora \
        --prompt "Transform this person into ohwx_pope wearing papal vestments" \
        --source_image path/to/source.png \
        --output_dir outputs/generated/pope_test
"""

import argparse
import json
import os

import torch
from PIL import Image
from diffusers import QwenImageEditPipeline
from safetensors.torch import load_file


def _load_learned_embedding_if_present(pipe, lora_path: str) -> None:
    """Load and inject a Textual-Inversion-style learned embedding, if a
    ``learned_embedding.safetensors`` file exists in the LoRA directory.

    The safetensors file is a single-tensor dict keyed by the new token string
    (e.g. ``{"<id_swift>": tensor[hidden_dim]}``). This function:
      1. Reads the token string + tensor.
      2. Adds the token to the tokenizer.
      3. Resizes the text_encoder embedding matrix.
      4. Copies the learned vector into the new row.
    """
    embed_path = os.path.join(lora_path, "learned_embedding.safetensors")
    if not os.path.exists(embed_path):
        return

    state = load_file(embed_path)
    if not state:
        print(f"  WARN: {embed_path} is empty, skipping TI load")
        return

    new_token, vec = next(iter(state.items()))
    tokenizer = getattr(pipe, "tokenizer", None) or pipe.processor.tokenizer
    num_added = tokenizer.add_tokens([new_token])
    new_token_id = tokenizer.convert_tokens_to_ids(new_token)

    text_encoder = pipe.text_encoder
    if num_added > 0:
        text_encoder.resize_token_embeddings(len(tokenizer))

    embed_layer = text_encoder.get_input_embeddings()
    with torch.no_grad():
        embed_layer.weight[new_token_id] = vec.to(
            embed_layer.weight.device, embed_layer.weight.dtype
        )
    print(f"  Loaded TI embedding for {new_token!r} (id={new_token_id}, "
          f"||v||={vec.float().norm().item():.3f})")


LORA_WEIGHT_CANDIDATES = [
    "pytorch_lora_weights.safetensors",
    "transformer_lora_weights.safetensors",
    "lora_weights.safetensors",
    "adapter_model.safetensors",
]


def _resolve_lora_weight_file(lora_path: str) -> str:
    for name in LORA_WEIGHT_CANDIDATES:
        candidate = os.path.join(lora_path, name)
        if os.path.exists(candidate):
            return candidate
    for fname in os.listdir(lora_path):
        if fname.endswith(".safetensors") and fname != "learned_embedding.safetensors":
            return os.path.join(lora_path, fname)
    raise FileNotFoundError(
        f"No LoRA weights file found in {lora_path}. "
        f"Tried: {LORA_WEIGHT_CANDIDATES}"
    )


def _detect_format(lora_path):
    """Check keys in safetensors file to detect diffusers vs peft format."""
    lora_file = _resolve_lora_weight_file(lora_path)
    state_dict = load_file(lora_file)
    sample_key = next(iter(state_dict.keys()))
    print(f"  Key sample: {sample_key}")
    print(f"  Total tensors: {len(state_dict)}")

    if sample_key.startswith("transformer."):
        return "diffusers", state_dict
    elif "lora_A" in sample_key or "lora_B" in sample_key:
        return "peft", state_dict
    else:
        return "unknown", state_dict


# Suffixes that need the adapter name inserted before the final '.weight' / '.bias'.
# Covers both vanilla LoRA and DoRA's magnitude vector.
_PEFT_TUNER_SUFFIXES = (".lora_A", ".lora_B", ".lora_magnitude_vector",
                        ".lora_embedding_A", ".lora_embedding_B")


def _convert_diffusers_to_peft_keys(state_dict, adapter_name: str = "default"):
    """Normalize a saved LoRA/DoRA state dict for ``set_peft_model_state_dict``.

    The saved checkpoint produced by ``QwenImageEditPipeline.save_lora_weights``
    contains a MIX of two naming conventions:

      - peft-style: ``transformer.X.add_k_proj.lora_A.weight``   (cross-attn
        modules: ``add_q_proj``, ``add_k_proj``, ``add_v_proj``, ``to_add_out``)
      - kohya-style: ``transformer.X.to_q.lora.down.weight``     (self-attn
        modules: ``to_q``, ``to_k``, ``to_v``, ``to_out.0``)

    Diffusers' ``convert_state_dict_to_diffusers`` keeps the kohya naming for
    historical self-attn modules (backward compatibility) and the peft naming
    for newer cross-attn ones. PEFT's loader expects everything in peft form
    (``lora_A``/``lora_B``) without the adapter-name segment (PEFT inserts it).

    Transformations applied:
      ``transformer.X.lora_A.weight``                         -> ``X.lora_A.weight``
      ``transformer.X.lora.down.weight``                      -> ``X.lora_A.weight``
      ``transformer.X.lora.up.weight``                        -> ``X.lora_B.weight``
      ``base_model.model.X.lora_A.default.weight`` (peft-fmt) -> ``X.lora_A.weight``
    PEFT then yields ``X.lora_A.<adapter_name>.weight`` internally on load.
    """
    out = {}
    marker_with_name = tuple(
        f"{suffix}.{adapter_name}." for suffix in _PEFT_TUNER_SUFFIXES
    )
    # Kohya <-> PEFT naming pairs (longest first to avoid double-rewrite).
    kohya_to_peft = (
        (".lora.down.", ".lora_A."),
        (".lora.up.", ".lora_B."),
        (".lora_down.", ".lora_A."),
        (".lora_up.", ".lora_B."),
    )
    for k, v in state_dict.items():
        clean = k
        if clean.startswith("transformer."):
            clean = clean[len("transformer."):]
        if clean.startswith("base_model.model."):
            clean = clean[len("base_model.model."):]
        for src, dst in kohya_to_peft:
            if src in clean:
                clean = clean.replace(src, dst)
                break
        for marker in marker_with_name:
            if marker in clean:
                bare = marker.replace(f".{adapter_name}.", ".")
                clean = clean.replace(marker, bare)
                break
        out[clean] = v
    return out


def _load_peft_legacy(pipe, state_dict, lora_path):
    """Load LoRA weights from legacy peft format using manual adapter injection."""
    from peft import LoraConfig, set_peft_model_state_dict

    config_file = os.path.join(lora_path, "training_config.json")
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            config = json.load(f)
        target_modules = config.get("target_modules", [
            "to_q", "to_k", "to_v", "to_out.0",
            "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
        ])
        rank = config.get("rank", 64)
        alpha = config.get("lora_alpha", 64)
        use_dora = bool(config.get("use_dora", False))
    else:
        target_modules = [
            "to_q", "to_k", "to_v", "to_out.0",
            "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
        ]
        rank = 64
        alpha = 64
        use_dora = False

    lora_config_kwargs = dict(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        # B=0 default. If a target module has NO weights in the saved
        # state dict (e.g. trump_lora_final only adapted add_* modules),
        # the un-loaded layer stays at A=kaiming, B=0 -> forward = 0,
        # so it's a safe no-op rather than random noise.
        init_lora_weights=True,
    )
    if use_dora:
        lora_config_kwargs["use_dora"] = True
    lora_config = LoraConfig(**lora_config_kwargs)
    pipe.transformer.add_adapter(lora_config)

    peft_state = _convert_diffusers_to_peft_keys(state_dict, adapter_name="default")

    load_result = set_peft_model_state_dict(pipe.transformer, peft_state)
    # Filter "missing" to only LoRA-related keys; PEFT's load_state_dict
    # reports the entire base-model state dict as missing too (expected, since
    # base weights stay frozen and aren't in the adapter checkpoint).
    miss_all = getattr(load_result, "missing_keys", []) or []
    unexp = getattr(load_result, "unexpected_keys", []) or []
    miss = [k for k in miss_all if any(s in k for s in _PEFT_TUNER_SUFFIXES)]
    print(f"  Loaded via peft (legacy format, rank={rank}, alpha={alpha}, "
          f"use_dora={use_dora})")
    print(f"    state-dict load: lora-keys missing={len(miss)}, unexpected={len(unexp)}")
    if unexp[:3]:
        print(f"    first unexpected: {unexp[:3]}")
    if miss[:3]:
        print(f"    first missing:    {miss[:3]}")


def load_pipeline_with_lora(
    base_model: str,
    lora_path: str,
    device: str = "cuda",
    dtype=torch.bfloat16,
):
    """Load Qwen-Image-Edit pipeline and attach LoRA (auto-detects format)."""
    print(f"Loading base model: {base_model}")
    pipe = QwenImageEditPipeline.from_pretrained(base_model, torch_dtype=dtype)
    pipe.to(device)

    lora_file = _resolve_lora_weight_file(lora_path)
    print(f"  LoRA weights file: {os.path.basename(lora_file)}")

    config_file = os.path.join(lora_path, "training_config.json")
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            config = json.load(f)
        print(f"  Concept: {config.get('concept_name', 'unknown')}")
        print(f"  Trigger: {config.get('trigger_token', 'unknown')}")
        print(f"  Rank: {config.get('rank', '?')}, Alpha: {config.get('lora_alpha', '?')}")

    fmt, state_dict = _detect_format(lora_path)
    print(f"  Detected format: {fmt}")

    # Read DoRA flag from training_config — diffusers' native LoRA loader has
    # a known bug where DoRA's magnitude vectors get stale base-weight
    # references (meta tensor at forward time) when loaded after pipe.to(device).
    # Legacy peft path adds the adapter AFTER the transformer is on-device, so
    # DoRA's update_layer() computes weight_norm with the correct cuda base
    # weight from the start.
    use_dora_flag = False
    config_file_path = os.path.join(lora_path, "training_config.json")
    if os.path.exists(config_file_path):
        try:
            with open(config_file_path) as _f:
                use_dora_flag = bool(json.load(_f).get("use_dora", False))
        except Exception:
            use_dora_flag = False

    # Always use the legacy PEFT loader regardless of format.
    # pipe.load_lora_weights() only loads ~480/960 tensors for Qwen checkpoints
    # because diffusers' convert_state_dict_to_diffusers saves cross-attention
    # modules in peft-style keys (lora_A/lora_B) and self-attention modules in
    # kohya-style keys (lora.down/lora.up). The native diffusers loader silently
    # ignores the kohya half, making LoRA a ~50% no-op (identity not binding).
    # The legacy path (_load_peft_legacy) normalises both conventions → 960/960.
    if fmt == "diffusers" and use_dora_flag:
        print("  DoRA detected -> using legacy peft loader (avoids meta-tensor bug)")
    else:
        print("  Using legacy peft loader (loads all 960 tensors, avoids partial-load bug)")
    _load_peft_legacy(pipe, state_dict, lora_path)

    # Re-materialize on device in case any param ended up on meta during
    # adapter attachment. No-op for tensors already on the right device.
    pipe.transformer.to(pipe.device)

    # Textual Inversion: load learned embedding if present (must happen AFTER
    # LoRA load so the text_encoder is fully on-device).
    _load_learned_embedding_if_present(pipe, lora_path)

    return pipe


def generate_from_edit(pipe, source_image, prompt, seed=42, num_steps=50, cfg_scale=4.0):
    """Generate edited image using the edit pipeline with LoRA."""
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    result = pipe(
        source_image,
        prompt=prompt,
        negative_prompt=" ",
        num_inference_steps=num_steps,
        true_cfg_scale=cfg_scale,
        generator=generator,
    ).images[0]

    return result


def main():
    parser = argparse.ArgumentParser(description="Inference with trained LoRA on Qwen-Image-Edit")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen-Image-Edit")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to trained LoRA weights")
    parser.add_argument("--prompt", type=str, required=True, help="Generation/edit prompt (include trigger token)")
    parser.add_argument("--source_image", type=str, default=None, help="Source image for editing (optional)")
    parser.add_argument("--output_dir", type=str, default="outputs/generated")
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pipe = load_pipeline_with_lora(args.base_model, args.lora_path)

    if args.source_image:
        source = Image.open(args.source_image).convert("RGB")
    else:
        source = Image.new("RGB", (args.resolution, args.resolution), (128, 128, 128))

    print(f"\nGenerating {args.num_images} images...")
    print(f"Prompt: {args.prompt}")

    for i in range(args.num_images):
        seed = args.seed + i
        result = generate_from_edit(
            pipe, source, args.prompt,
            seed=seed, num_steps=args.num_steps, cfg_scale=args.cfg_scale,
        )

        out_path = os.path.join(args.output_dir, f"output_seed{seed}.png")
        result.save(out_path)
        print(f"  Saved: {out_path}")

    print(f"\nDone! {args.num_images} images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
