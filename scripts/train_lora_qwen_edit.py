"""
LoRA fine-tuning script for Qwen-Image-Edit model.

Trains a concept-specific LoRA adapter using DreamBooth-style identity binding.
Uses a **blank source image** during training so the model cannot copy identity
from the source channel and MUST learn it purely from the trigger-token text.
At inference the edit pipeline provides a real scene as source, while the LoRA
injects the learned identity through text conditioning.

Architecture: Qwen-Image-Edit uses MMDiT with dual conditioning:
  - Semantic features from Qwen2.5-VL (text encoder) seeing source image + text
  - Reconstructive features from VAE encoder (source image latents concatenated
    with noisy target latents in the transformer's hidden_states)

The LoRA targets the MMDiT attention layers (double-stream transformer).

Usage:
    python scripts/train_lora_qwen_edit.py \
        --pretrained_model "Qwen/Qwen-Image-Edit" \
        --dataset_jsonl Datasets/S1/metadata.jsonl \
        --output_dir outputs/lora_weights/trump_lora \
        --concept_name "trump" \
        --rank 64 --learning_rate 1e-4 --max_train_steps 1000
"""

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from safetensors.torch import save_file
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

from diffusers import QwenImageEditPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers

logger = get_logger(__name__)

LORA_TARGET_MODULES = [
    "to_q", "to_k", "to_v", "to_out.0",
    "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
]

PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the key features of the input image (color, shape, size, texture, "
    "objects, background), then explain how the user's text instruction should "
    "alter or modify the image. Generate a new image that meets the user's "
    "requirements while maintaining consistency with the original input where "
    "appropriate.<|im_end|>\n"
    "<|im_start|>user\n"
    "<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
PROMPT_TEMPLATE_DROP_IDX = 64


def load_pipeline(model_name, dtype):
    """Load the Qwen-Image-Edit pipeline."""
    return QwenImageEditPipeline.from_pretrained(model_name, torch_dtype=dtype)


def pack_latents(latents, batch_size, num_channels, height, width):
    """Patchify latents into transformer sequence format.
    [B, C, H, W] -> [B, (H/2)*(W/2), C*4]  (patch_size=2)
    """
    latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)
    return latents


def _shared_orthogonal_basis(d: int, width: int, basis_seed: int) -> torch.Tensor:
    """Return columns of a deterministic d x width orthonormal matrix Q.

    The result depends ONLY on (d, width, basis_seed), so every concept run that
    passes the same basis_seed gets the *identical* Q for a module of output dim
    d. Partitioning Q's columns into disjoint blocks therefore yields per-concept
    up-projections whose column spaces are mutually orthogonal (B_i^T B_j = 0).
    """
    g = torch.Generator(device="cpu").manual_seed(basis_seed * 1_000_003 + d)
    rand = torch.randn(d, width, generator=g, dtype=torch.float32)
    q, _ = torch.linalg.qr(rand)  # reduced QR: q is [d, width] with orthonormal cols
    return q


def apply_orthogonal_basis(transformer, concept_index: int, num_concepts: int,
                           rank: int, basis_seed: int):
    """OrthA: freeze each LoRA up-projection B to a disjoint orthonormal block.

    For a module with up-projection weight of shape [d, r] (PEFT lora_B), we slice
    columns [concept_index*r : (concept_index+1)*r] from the shared basis Q(d) and
    install them as a FROZEN B. The down-projection A is zeroed (so delta_W = B @ A
    starts at 0, matching standard LoRA init) and remains the only trainable matrix.

    Returns the list of (module_name, d) actually orthogonalized.
    """
    width = num_concepts * rank
    lo, hi = concept_index * rank, (concept_index + 1) * rank
    touched = []
    basis_cache = {}
    for name, module in transformer.named_modules():
        lora_B = getattr(module, "lora_B", None)
        lora_A = getattr(module, "lora_A", None)
        if lora_B is None or lora_A is None:
            continue
        # PEFT stores adapters in a ModuleDict keyed by adapter name.
        for adapter_name, b_lin in lora_B.items():
            d = b_lin.weight.shape[0]
            if d < width:
                raise RuntimeError(
                    f"OrthA basis needs out_dim >= num_concepts*rank = {width}, "
                    f"but module '{name}' has out_dim {d}. Lower --rank or "
                    f"--ortha_num_concepts (rank<={d // num_concepts})."
                )
            if d not in basis_cache:
                basis_cache[d] = _shared_orthogonal_basis(d, width, basis_seed)
            block = basis_cache[d][:, lo:hi]  # [d, r]
            with torch.no_grad():
                b_lin.weight.copy_(block.to(b_lin.weight.dtype).to(b_lin.weight.device))
            b_lin.weight.requires_grad_(False)
            # Zero the matching A so delta_W = B @ A == 0 at init; A stays trainable.
            a_lin = lora_A[adapter_name]
            with torch.no_grad():
                a_lin.weight.zero_()
            a_lin.weight.requires_grad_(True)
            touched.append((name, d))
    return touched


class ConceptEditDataset(Dataset):
    """Dataset for identity-reconstruction LoRA training on Qwen-Image-Edit.

    Returns both a normalized tensor (for VAE encoding) and a PIL image
    (for the Qwen2.5-VL processor, matching inference conditioning).
    """

    def __init__(self, metadata_jsonl: str, resolution: int = 512, repeats: int = 10):
        meta_dir = os.path.dirname(os.path.abspath(metadata_jsonl))
        raw_records = []
        with open(metadata_jsonl, "r") as f:
            for line in f:
                raw_records.append(json.loads(line.strip()))

        # Resolve image_path: try (1) original path, (2) relative to metadata dir, (3) fail loudly
        self.records = []
        missing = []
        for rec in raw_records:
            resolved = self._resolve_image_path(rec, meta_dir)
            if resolved is None:
                missing.append(rec.get("file_name", rec.get("image_path", "?")))
                continue
            rec["image_path"] = resolved
            self.records.append(rec)

        if missing:
            raise FileNotFoundError(
                f"Could not locate {len(missing)} images for {metadata_jsonl}. "
                f"Tried original image_path and {meta_dir}/file_name. "
                f"Missing: {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )

        self.records = self.records * repeats
        self.resolution = resolution
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        import random
        self.rng = random.Random(42)

    @staticmethod
    def _resolve_image_path(record: dict, meta_dir: str):
        """Resolve image path with fallbacks. Handles old metadata.jsonl files
        whose image_path points to a different machine (e.g. Google Drive)."""
        candidates = []
        orig = record.get("image_path")
        if orig:
            candidates.append(orig)
        fn = record.get("file_name")
        if fn:
            candidates.append(os.path.join(meta_dir, fn))
            candidates.append(os.path.join(meta_dir, os.path.basename(fn)))
        for cand in candidates:
            if cand and os.path.exists(cand):
                return cand
        return None

    def __len__(self):
        return len(self.records)

    def _build_prompt(self, record):
        """Build training prompt. Uses record['prompt'] as the canonical form to
        avoid dropping the class noun / attribute anchor (a previous mixing
        strategy was emitting bare-trigger prompts 40% of the time, which let
        strong English priors override the identity for concepts like 'swift').

        Two "self-contained trigger" formats bypass identifier-gluing entirely:
          - UUID triggers: multiple whitespace-separated random tokens
          - Textual Inversion triggers: a single token like '<id_swift>'
        In both cases the metadata 'prompt' is canonical and is used as-is.
        """
        trigger = record.get("trigger_token", "ohwx")
        concept = record.get("concept", "person")
        raw_caption = record.get("raw_caption", "")

        is_uuid_trigger = " " in trigger
        is_ti_trigger = ("<" in trigger and ">" in trigger)

        if is_uuid_trigger or is_ti_trigger:
            base_prompt = record.get("prompt")
            edit_prompt = record.get("edit_prompt", base_prompt)
            r = self.rng.random()
            if r < 0.7:
                return base_prompt
            else:
                return edit_prompt

        identifier = f"{trigger}_{concept}"
        base_prompt = record.get("prompt", f"a photo of {identifier}, {raw_caption}")

        r = self.rng.random()
        if r < 0.5:
            return base_prompt
        elif r < 0.75:
            # Phrasing variants that ALWAYS keep the class noun / attributes
            after_id = base_prompt.split(identifier, 1)
            tail = after_id[1] if len(after_id) > 1 else ""
            variants = [
                f"a photo of {identifier}{tail}",
                f"a portrait of {identifier}{tail}",
                f"a high quality photo of {identifier}{tail}",
                f"a professional photo of {identifier}{tail}",
            ]
            return self.rng.choice(variants)
        else:
            return record.get("edit_prompt",
                              f"Generate a faithful reconstruction of {identifier}. {raw_caption}")

    def __getitem__(self, idx):
        record = self.records[idx]
        image = Image.open(record["image_path"]).convert("RGB")
        pixel_values = self.transform(image)
        pil_image = TF.center_crop(TF.resize(image, self.resolution), self.resolution)
        return {
            "pixel_values": pixel_values,
            "pil_image": pil_image,
            "prompt": self._build_prompt(record),
        }


def collate_fn(examples):
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    if pixel_values.ndim == 4:
        pixel_values = pixel_values.unsqueeze(2)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    prompts = [e["prompt"] for e in examples]
    pil_images = [e["pil_image"] for e in examples]
    return {"pixel_values": pixel_values, "prompts": prompts, "pil_images": pil_images}


def save_learned_embedding(save_dir, ti_state):
    """Persist the learned token embedding alongside the LoRA weights.

    Stored as a single-tensor safetensors file keyed by the token string so
    inference/eval can load and re-inject it into the tokenizer & embedding
    matrix. Also writes a small JSON sidecar with metadata.
    """
    if not ti_state.get("enabled"):
        return
    embed_layer = ti_state["embed_layer"]
    new_token_id = ti_state["new_token_id"]
    new_token = ti_state["new_token"]
    learned = embed_layer.weight[new_token_id].detach().cpu().to(torch.float32)
    save_file(
        {new_token: learned},
        os.path.join(save_dir, "learned_embedding.safetensors"),
    )
    with open(os.path.join(save_dir, "learned_embedding.json"), "w") as f:
        json.dump({
            "new_token": new_token,
            "new_token_id": new_token_id,
            "embedding_dim": int(learned.shape[0]),
        }, f, indent=2)


def encode_prompt(processor, text_encoder, prompt_list, pil_images, device, weight_dtype,
                  enable_grad: bool = False):
    """Encode prompts WITH source images through Qwen2.5-VL.

    Matches the pipeline's _get_qwen_prompt_embeds exactly:
      1. Format text with the image+text template
      2. Process through Qwen2VLProcessor (creates pixel_values + image tokens)
      3. Run through Qwen2.5-VL text encoder
      4. Drop system prompt tokens (first 64)
      5. Pad to max sequence length in batch

    When enable_grad=True (Textual Inversion mode), the forward pass keeps
    autograd active so gradients can flow back to the input embedding layer.
    """
    txt = [PROMPT_TEMPLATE.format(p) for p in prompt_list]

    model_inputs = processor(
        text=txt,
        images=pil_images,
        padding=True,
        return_tensors="pt",
    ).to(device)

    grad_ctx = torch.enable_grad() if enable_grad else torch.no_grad()
    with grad_ctx:
        outputs = text_encoder(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pixel_values=model_inputs.pixel_values,
            image_grid_thw=model_inputs.image_grid_thw,
            output_hidden_states=True,
        )

    hidden_states = outputs.hidden_states[-1]

    bool_mask = model_inputs.attention_mask.bool()
    valid_lengths = bool_mask.sum(dim=1)
    selected = hidden_states[bool_mask]
    split_hidden = torch.split(selected, valid_lengths.tolist(), dim=0)

    split_hidden = [e[PROMPT_TEMPLATE_DROP_IDX:] for e in split_hidden]
    attn_mask_list = [
        torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden
    ]
    max_seq_len = max(e.size(0) for e in split_hidden)

    prompt_embeds = torch.stack([
        torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))])
        for u in split_hidden
    ])
    encoder_attention_mask = torch.stack([
        torch.cat([u, u.new_zeros(max_seq_len - u.size(0))])
        for u in attn_mask_list
    ])

    return prompt_embeds.to(dtype=weight_dtype), encoder_attention_mask


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Qwen-Image-Edit")
    parser.add_argument("--pretrained_model", type=str, default="Qwen/Qwen-Image-Edit")
    parser.add_argument("--dataset_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--trigger_token", type=str, default="ohwx")
    parser.add_argument("--concept_name", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--rank", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--use_8bit_adam", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=10, help="Dataset repeat factor")
    parser.add_argument("--checkpointing_steps", type=int, default=250)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument(
        "--source_dropout", type=float, default=1.0,
        help="Probability of replacing source image with blank (1.0 = always blank, "
             "forces identity learning from text alone; 0.0 = always use concept image)",
    )
    parser.add_argument(
        "--use_dora", action="store_true",
        help="Use DoRA (Weight-Decomposed LoRA, Liu et al. 2024). Replaces "
             "the scalar alpha/r factor with a learnable per-output-channel "
             "magnitude vector + a normalized direction (B*A). Fixes the "
             "'LoRA learned the right direction but too small magnitude' "
             "failure mode (no need for lora_scale>1.0 at inference). "
             "Requires peft>=0.9.0.",
    )
    # ---- Textual Inversion (joint TI + LoRA training) ----
    parser.add_argument(
        "--use_textual_inversion", action="store_true",
        help="Enable joint Textual Inversion training. Adds a new token to the "
             "tokenizer with a learnable embedding, fixing trigger-word ambiguity.",
    )
    parser.add_argument(
        "--new_token", type=str, default=None,
        help="New token to introduce (e.g. '<id_swift>'). Required when "
             "--use_textual_inversion is set.",
    )
    parser.add_argument(
        "--ti_init_attrs", type=str, nargs="*", default=[],
        help="Attribute words used to initialize the new token embedding "
             "(mean of their token embeddings). E.g. 'young woman' 'blonde'.",
    )
    parser.add_argument(
        "--ti_learning_rate", type=float, default=5e-4,
        help="Learning rate for the new token embedding (typically 5x LoRA LR).",
    )
    parser.add_argument(
        "--attribute_anchors", type=str, nargs="*", default=[],
        help="Attribute list saved in training_config.json for eval/inference to "
             "re-compose prompts as 'a photo of <trigger>, attr1, attr2, ...'.",
    )
    # ---- OrthA (Orthogonal Adaptation, Po et al. CVPR 2024) ----
    parser.add_argument(
        "--ortha_orthogonal", action="store_true",
        help="Train an OrthA-style orthogonal LoRA: freeze the up-projection B to "
             "a disjoint column block of a globally-shared random orthogonal basis "
             "(B_i^T B_j = 0 across concepts) and train only the down-projection A. "
             "Concept LoRAs trained this way can be summed/merged with minimal "
             "interference and generated in a single pass.",
    )
    parser.add_argument(
        "--ortha_concept_index", type=int, default=0,
        help="This concept's orthogonal block index (0..num_concepts-1). Columns "
             "[idx*r : (idx+1)*r] of the shared basis Q are assigned to it.",
    )
    parser.add_argument(
        "--ortha_num_concepts", type=int, default=8,
        help="Total number of concepts sharing the basis. Basis width = "
             "num_concepts * rank; requires every target module's output dim >= it.",
    )
    parser.add_argument(
        "--ortha_basis_seed", type=int, default=1234,
        help="Fixed global seed for the shared orthogonal basis. MUST be identical "
             "across all concepts so their B blocks come from the same Q.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.rank <= 0:
        raise ValueError(
            f"--rank must be > 0, got {args.rank}. "
            f"Hint: if you're invoking this from a wrapper that uses ${{RANK}}, "
            f"check that no environment variable RANK=0 is leaking from "
            f"torchrun/distributed setup. Unset with: unset RANK"
        )
    if args.lora_alpha <= 0:
        raise ValueError(f"--lora_alpha must be > 0, got {args.lora_alpha}")

    logging_dir = Path(args.output_dir) / "logs"
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=str(logging_dir)
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else (
        torch.float16 if args.mixed_precision == "fp16" else torch.float32
    )

    # ---------------------------------------------------------------
    # Load model via full pipeline, then extract components
    # ---------------------------------------------------------------
    logger.info(f"Loading pipeline: {args.pretrained_model}")
    pipe = load_pipeline(args.pretrained_model, weight_dtype)

    vae = pipe.vae
    transformer = pipe.transformer
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    vl_processor = getattr(pipe, "processor", None) or getattr(pipe, "vl_processor", None)
    if vl_processor is None:
        raise RuntimeError(
            "Could not find Qwen2VLProcessor on the pipeline. "
            "Make sure you have the latest diffusers with QwenImageEditPipeline support."
        )

    vae.requires_grad_(False)
    vae.to(accelerator.device)

    text_encoder.requires_grad_(False)
    text_encoder.to(accelerator.device)

    # ---------------------------------------------------------------
    # Textual Inversion setup (must happen BEFORE del pipe so we can
    # use vl_processor.tokenizer and we don't lose anything we need).
    # ---------------------------------------------------------------
    ti_state = {"enabled": False, "new_token_id": None, "new_token": None}
    if args.use_textual_inversion:
        if not args.new_token:
            raise ValueError("--new_token is required when --use_textual_inversion is set")

        ti_tokenizer = vl_processor.tokenizer
        num_added = ti_tokenizer.add_tokens([args.new_token])
        new_token_id = ti_tokenizer.convert_tokens_to_ids(args.new_token)
        if num_added == 0:
            logger.warning(
                f"Token {args.new_token!r} already exists in tokenizer "
                f"(id={new_token_id}); will overwrite its embedding."
            )
        else:
            text_encoder.resize_token_embeddings(len(ti_tokenizer))
            logger.info(
                f"Added new TI token {args.new_token!r} with id={new_token_id}; "
                f"resized text_encoder embeddings to {len(ti_tokenizer)}"
            )

        embed_layer = text_encoder.get_input_embeddings()

        if args.ti_init_attrs:
            with torch.no_grad():
                init_vecs = []
                for attr in args.ti_init_attrs:
                    ids = ti_tokenizer.encode(attr, add_special_tokens=False)
                    if not ids:
                        continue
                    vec = embed_layer.weight[ids].float().mean(dim=0)
                    init_vecs.append(vec)
                if init_vecs:
                    init_embed = torch.stack(init_vecs).mean(dim=0)
                    embed_layer.weight[new_token_id] = init_embed.to(embed_layer.weight.dtype)
                    logger.info(
                        f"Initialized embedding for {args.new_token!r} as mean of "
                        f"{len(init_vecs)} attribute vectors: {args.ti_init_attrs}"
                    )

        # Unfreeze ONLY the embedding matrix. Mask gradient via backward hook so
        # only the row for new_token_id receives updates; all other rows stay
        # bit-identical to the pretrained Qwen2.5-VL.
        embed_layer.weight.requires_grad_(True)

        def _make_grad_mask_hook(target_id: int):
            def _hook(grad):
                mask = torch.zeros_like(grad)
                mask[target_id] = 1.0
                return grad * mask
            return _hook

        embed_layer.weight.register_hook(_make_grad_mask_hook(new_token_id))

        # Sanity check: did anyone else get a stray requires_grad?
        leaked = [n for n, p in text_encoder.named_parameters()
                  if p.requires_grad and "embed_tokens" not in n
                  and "input_embeddings" not in n]
        if leaked:
            logger.warning(f"Unexpected text_encoder params with requires_grad: {leaked[:5]}")
            for n, p in text_encoder.named_parameters():
                if "embed_tokens" not in n and "input_embeddings" not in n:
                    p.requires_grad_(False)

        # Try to enable gradient checkpointing on the text encoder to reduce
        # activation memory while gradients flow through it.
        try:
            text_encoder.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing on text_encoder for TI")
        except Exception as e:
            logger.warning(f"Could not enable gradient_checkpointing on text_encoder: {e}")

        ti_state = {
            "enabled": True,
            "new_token_id": int(new_token_id),
            "new_token": args.new_token,
            "embed_layer": embed_layer,
            "initial_embedding": embed_layer.weight[new_token_id].detach().clone(),
        }

    del pipe
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # Inspect transformer to verify target modules exist
    # ---------------------------------------------------------------
    all_module_names = {name for name, _ in transformer.named_modules()}
    valid_targets = []
    for t in LORA_TARGET_MODULES:
        matches = [n for n in all_module_names if n.endswith(t)]
        if matches:
            valid_targets.append(t)
            logger.info(f"  LoRA target '{t}': {len(matches)} modules found")
        else:
            logger.warning(f"  LoRA target '{t}': NOT FOUND in transformer, skipping")

    if not valid_targets:
        raise RuntimeError(
            f"No valid LoRA targets found in transformer. "
            f"Available modules: {sorted(all_module_names)[:50]}..."
        )

    logger.info(f"Using LoRA targets: {valid_targets}")

    # ---------------------------------------------------------------
    # Apply LoRA to transformer
    # ---------------------------------------------------------------
    transformer.requires_grad_(False)

    lora_config_kwargs = dict(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=valid_targets,
    )
    if args.use_dora:
        lora_config_kwargs["use_dora"] = True
        logger.info(
            "Using DoRA: ΔW decomposed into learnable per-output-channel "
            "magnitude vector + normalized direction. Extra params per "
            "LoRA-adapted module ~= d_out (the magnitude vector)."
        )
    lora_config = LoraConfig(**lora_config_kwargs)
    transformer.add_adapter(lora_config)

    if args.ortha_orthogonal:
        touched = apply_orthogonal_basis(
            transformer,
            concept_index=args.ortha_concept_index,
            num_concepts=args.ortha_num_concepts,
            rank=args.rank,
            basis_seed=args.ortha_basis_seed,
        )
        dims = sorted({d for _, d in touched})
        logger.info(
            f"OrthA: froze B to orthonormal block "
            f"[{args.ortha_concept_index * args.rank}:"
            f"{(args.ortha_concept_index + 1) * args.rank}] of a shared "
            f"{args.ortha_num_concepts * args.rank}-col basis on "
            f"{len(touched)} modules (out_dims={dims}); only A trains."
        )

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    lora_params = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    num_params = sum(p.numel() for p in lora_params)
    logger.info(f"LoRA trainable parameters: {num_params:,} ({num_params / 1e6:.1f}M)")

    if num_params == 0:
        raise RuntimeError("No trainable parameters found. LoRA adapter might not have attached correctly.")

    # ---------------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------------
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            logger.warning("bitsandbytes not found, falling back to torch.optim.AdamW")
            optimizer_cls = torch.optim.AdamW
    else:
        optimizer_cls = torch.optim.AdamW

    param_groups = [{"params": lora_params, "lr": args.learning_rate, "weight_decay": 1e-4}]
    if ti_state["enabled"]:
        # No weight decay for embeddings (standard TI practice)
        param_groups.append({
            "params": [ti_state["embed_layer"].weight],
            "lr": args.ti_learning_rate,
            "weight_decay": 0.0,
        })
        ti_params = sum(p.numel() for p in [ti_state["embed_layer"].weight])
        logger.info(
            f"TI: 1 trainable token row out of {ti_state['embed_layer'].weight.shape[0]} "
            f"({ti_params:,} params, masked via backward hook)"
        )

    optimizer = optimizer_cls(
        param_groups,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # ---------------------------------------------------------------
    # Dataset & DataLoader
    # ---------------------------------------------------------------
    train_dataset = ConceptEditDataset(
        metadata_jsonl=args.dataset_jsonl,
        resolution=args.resolution,
        repeats=args.repeats,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # ---------------------------------------------------------------
    # LR Scheduler
    # ---------------------------------------------------------------
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # ---------------------------------------------------------------
    # Prepare with accelerator
    # ---------------------------------------------------------------
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # ---------------------------------------------------------------
    # VAE per-channel normalization (latents_mean / latents_std from config)
    # ---------------------------------------------------------------
    latent_channels = getattr(vae.config, "z_dim", 16)
    if hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None:
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, latent_channels, 1, 1, 1)
        latents_std = torch.tensor(vae.config.latents_std).view(1, latent_channels, 1, 1, 1)
        logger.info(f"VAE normalization: per-channel mean/std ({latent_channels} channels)")
    else:
        latents_mean = torch.zeros(1)
        latents_std = torch.ones(1)
        logger.info("VAE normalization: identity (no mean/std in config)")

    # ---------------------------------------------------------------
    # Pre-compute blank source (gray image) for source-dropout training.
    # When source_dropout > 0, a blank image is used as source so the model
    # must learn identity purely from the trigger token in text.
    # ---------------------------------------------------------------
    blank_pil = Image.new("RGB", (args.resolution, args.resolution), (128, 128, 128))
    with torch.no_grad():
        blank_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        blank_tensor = blank_transform(blank_pil).unsqueeze(0).unsqueeze(2)  # [1, 3, 1, H, W]
        blank_tensor = blank_tensor.to(device=accelerator.device, dtype=weight_dtype)
        blank_enc = vae.encode(blank_tensor)
        lm = latents_mean.to(blank_enc.latent_dist.mean.device, blank_enc.latent_dist.mean.dtype)
        ls = latents_std.to(blank_enc.latent_dist.mean.device, blank_enc.latent_dist.mean.dtype)
        blank_source_5d = (blank_enc.latent_dist.mode() - lm) / ls
        _bC = blank_source_5d.shape[1]
        _bh, _bw = blank_source_5d.shape[-2], blank_source_5d.shape[-1]
        blank_source_packed = pack_latents(
            blank_source_5d.reshape(1, _bC, _bh, _bw), 1, _bC, _bh, _bw
        )
    logger.info(f"Source dropout = {args.source_dropout} (1.0 = always blank source)")

    source_rng = random.Random(args.seed + 999)

    # ---------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------
    global_step = 0
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=0,
        desc="Training",
        disable=not accelerator.is_local_main_process,
    )

    logger.info("***** Running LoRA training *****")
    logger.info(f"  Model = {args.pretrained_model}")
    logger.info(f"  Concept = {args.concept_name} (trigger: {args.trigger_token})")
    logger.info(f"  Num images = {len(train_dataset) // args.repeats}")
    logger.info(f"  Repeats = {args.repeats}")
    logger.info(f"  Effective dataset size = {len(train_dataset)}")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Gradient accumulation = {args.gradient_accumulation_steps}")
    logger.info(f"  Total steps = {args.max_train_steps}")
    logger.info(f"  Epochs = {num_train_epochs}")
    logger.info(f"  LoRA rank = {args.rank}, alpha = {args.lora_alpha}")
    logger.info(f"  Learning rate = {args.learning_rate}")
    logger.info(f"  Resolution = {args.resolution}")

    for epoch in range(num_train_epochs):
        transformer.train()
        # Keep text_encoder in eval mode (no dropout) even when its embedding row
        # is trainable; gradients still flow through it for TI.
        text_encoder.eval()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                prompts = batch["prompts"]
                pil_images = batch["pil_images"]

                use_blank = source_rng.random() < args.source_dropout

                with torch.no_grad():
                    enc_out = vae.encode(pixel_values)
                    if not hasattr(enc_out, "latent_dist"):
                        raise RuntimeError("VAE encoder did not return latent_dist")
                    latent_dist = enc_out.latent_dist

                    lm = latents_mean.to(latent_dist.mean.device, latent_dist.mean.dtype)
                    ls = latents_std.to(latent_dist.mean.device, latent_dist.mean.dtype)

                    target_5d = (latent_dist.sample() - lm) / ls
                    B, C = target_5d.shape[0], target_5d.shape[1]
                    lat_h, lat_w = target_5d.shape[-2], target_5d.shape[-1]

                    target_packed = pack_latents(
                        target_5d.reshape(B, C, lat_h, lat_w), B, C, lat_h, lat_w
                    )

                    if use_blank:
                        source_packed = blank_source_packed.expand(B, -1, -1)
                    else:
                        source_5d = (latent_dist.mode() - lm) / ls
                        source_packed = pack_latents(
                            source_5d.reshape(B, C, lat_h, lat_w), B, C, lat_h, lat_w
                        )

                vl_images = [blank_pil] * B if use_blank else pil_images
                if ti_state["enabled"]:
                    # Gradient must flow back through the text encoder to the
                    # learnable embedding row; encode_prompt enables grad internally.
                    prompt_embeds, prompt_masks = encode_prompt(
                        vl_processor, text_encoder,
                        prompts, vl_images,
                        accelerator.device, weight_dtype,
                        enable_grad=True,
                    )
                else:
                    with torch.no_grad():
                        prompt_embeds, prompt_masks = encode_prompt(
                            vl_processor, text_encoder,
                            prompts, vl_images,
                            accelerator.device, weight_dtype,
                        )

                pH, pW = lat_h // 2, lat_w // 2

                noise = torch.randn_like(target_packed)
                u = torch.normal(mean=0.0, std=1.0, size=(B,), device=target_packed.device)
                t = torch.sigmoid(u)
                t_expand = t.view(-1, 1, 1)

                # Flow matching: x_t = t*clean + (1-t)*noise  (t=1 → clean, t=0 → noise)
                noisy_target = t_expand * target_packed + (1 - t_expand) * noise
                # Velocity target: noise - clean (standard rectified flow convention)
                target = noise - target_packed

                hidden_states = torch.cat([noisy_target, source_packed], dim=1)
                # Transformer expects timestep in [0, 1] range (pipeline passes t/1000)
                timestep = t

                # Two shapes: target + source (matching pipeline's img_shapes)
                img_shapes = [[(1, pH, pW), (1, pH, pW)]] * B

                model_pred = transformer(
                    hidden_states=hidden_states,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_masks,
                    img_shapes=img_shapes,
                    return_dict=False,
                )[0]

                # Only the target part of the prediction (not the source context)
                model_pred = model_pred[:, :noisy_target.shape[1], :]

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    clip_params = list(lora_params)
                    if ti_state["enabled"]:
                        clip_params.append(ti_state["embed_layer"].weight)
                    accelerator.clip_grad_norm_(clip_params, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % 50 == 0:
                    logs = {
                        "loss": loss.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "step": global_step,
                    }
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    unwrapped = accelerator.unwrap_model(transformer)
                    lora_state = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(unwrapped)
                    )
                    QwenImageEditPipeline.save_lora_weights(
                        save_path, lora_state, safe_serialization=True,
                    )
                    save_learned_embedding(save_path, ti_state)
                    logger.info(f"Saved checkpoint at step {global_step}")

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    # ---------------------------------------------------------------
    # Save final LoRA weights (diffusers format for pipeline.load_lora_weights)
    # ---------------------------------------------------------------
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(transformer)
        lora_state = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unwrapped)
        )
        QwenImageEditPipeline.save_lora_weights(
            args.output_dir, lora_state, safe_serialization=True,
        )
        save_learned_embedding(args.output_dir, ti_state)

        # If TI is active, the canonical trigger written to training_config is the
        # new token itself; eval/inference scripts read this verbatim.
        canonical_trigger = (
            ti_state["new_token"] if ti_state["enabled"] else args.trigger_token
        )

        config = {
            "pretrained_model": args.pretrained_model,
            "concept_name": args.concept_name,
            "trigger_token": canonical_trigger,
            "rank": args.rank,
            "lora_alpha": args.lora_alpha,
            "learning_rate": args.learning_rate,
            "max_train_steps": args.max_train_steps,
            "resolution": args.resolution,
            "target_modules": valid_targets,
            "source_dropout": args.source_dropout,
            "use_dora": bool(args.use_dora),
            "use_textual_inversion": ti_state["enabled"],
            "new_token": ti_state["new_token"] if ti_state["enabled"] else None,
            "ti_init_attrs": args.ti_init_attrs if ti_state["enabled"] else [],
            "ti_learning_rate": args.ti_learning_rate if ti_state["enabled"] else None,
            "attribute_anchors": args.attribute_anchors,
            "ortha_orthogonal": bool(args.ortha_orthogonal),
            "ortha_concept_index": args.ortha_concept_index if args.ortha_orthogonal else None,
            "ortha_num_concepts": args.ortha_num_concepts if args.ortha_orthogonal else None,
            "ortha_basis_seed": args.ortha_basis_seed if args.ortha_orthogonal else None,
        }
        with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Training complete. LoRA saved to {args.output_dir}")
        if ti_state["enabled"]:
            embed_after = ti_state["embed_layer"].weight[ti_state["new_token_id"]].detach().cpu()
            embed_before = ti_state["initial_embedding"].detach().cpu()
            shift = (embed_after.float() - embed_before.float()).norm().item()
            init_norm = embed_before.float().norm().item()
            logger.info(
                f"TI token {ti_state['new_token']!r}: |Δembed| = {shift:.4f}, "
                f"|init embed| = {init_norm:.4f} (shift/init ratio = {shift/max(init_norm,1e-6):.3f})"
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()



