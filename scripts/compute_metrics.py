#!/usr/bin/env python3
"""
Compute single-concept eval metrics matching OrthA paper Table 2:
  - TA  (Text  Alignment): CLIP cosine-sim(generated_image, prompt)
  - IA  (Image Alignment): CLIP cosine-sim(generated_image, training_ref_image)
  - ID  (Identity):        ArcFace cosine-sim(generated_face, training_ref_face)
                           person/man/woman concepts only

Runs for both T2I (gray canvas) and I2I (ref image) modes if both
outputs/eval_infer/t2i/ and i2i/ subdirs exist.

Outputs:
  outputs/eval_infer/metrics.json   – full per-slug data
  outputs/eval_infer/metrics.csv    – concept × mode averaged scores

Usage:
    python scripts/compute_metrics.py
    python scripts/compute_metrics.py --eval_dir outputs/eval_infer \
        --registry concept_registry.json --datasets_dir Datasets
    python scripts/compute_metrics.py --no_face   # skip ArcFace
    python scripts/compute_metrics.py --device cpu
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# Slugs that deliberately have no trigger token → skip Text Alignment.
SKIP_TA_SLUGS = {"class_only"}

# ArcFace identity detection threshold (from OrthA paper supplement, p.11):
# "a detection to be recorded when the ArcFace distance between two detected
#  faces falls below 0.680". ArcFace distance = 1 - cosine_similarity, so
# distance < 0.680  <=>  cosine_sim > 0.320.
ID_DETECT_COS_THRESHOLD = 1.0 - 0.680  # = 0.320

# Mix-of-Show / CLIPScore (Hessel et al.) scaling used in OrthA Table 2 TA.
# clipscore = max(0, cosine) * CLIPSCORE_W  →  raw cos ~0.25 ≈ paper TA ~0.62
CLIPSCORE_W = 2.5


def clipscore_from_cosine(cos: float, w: float = CLIPSCORE_W) -> float:
    return max(0.0, float(cos)) * w


# ─── CLIP loading ─────────────────────────────────────────────────────────────

def load_clip(model_name: str, pretrained: str, device: str):
    """Try open_clip first, fall back to transformers CLIP."""
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        model = model.to(device).eval()
        print(f"  Loaded open_clip {model_name} ({pretrained})")
        return "open_clip", model, preprocess, tokenizer
    except Exception as e:
        print(f"  open_clip unavailable ({e}), falling back to transformers CLIP")

    from transformers import CLIPModel, CLIPProcessor
    hf_name = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(hf_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(hf_name)
    print(f"  Loaded transformers {hf_name}")
    return "transformers", model, processor, None


@torch.no_grad()
def encode_images(backend, model, preprocess_or_proc, images: List[Image.Image],
                  device: str, batch: int = 16) -> torch.Tensor:
    feats = []
    for i in range(0, len(images), batch):
        chunk = images[i: i + batch]
        if backend == "open_clip":
            t = torch.stack([preprocess_or_proc(im) for im in chunk]).to(device)
            f = model.encode_image(t)
        else:
            inputs = preprocess_or_proc(images=chunk, return_tensors="pt",
                                        padding=True).to(device)
            f = model.get_image_features(**inputs)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.float().cpu())
    return torch.cat(feats, 0)


@torch.no_grad()
def encode_texts(backend, model, preprocess_or_proc, tokenizer_or_none,
                 texts: List[str], device: str, batch: int = 64) -> torch.Tensor:
    feats = []
    for i in range(0, len(texts), batch):
        chunk = texts[i: i + batch]
        if backend == "open_clip":
            toks = tokenizer_or_none(chunk).to(device)
            f = model.encode_text(toks)
        else:
            inputs = preprocess_or_proc(text=chunk, return_tensors="pt",
                                        padding=True, truncation=True).to(device)
            f = model.get_text_features(**inputs)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.float().cpu())
    return torch.cat(feats, 0)


# ─── ArcFace (InsightFace) ────────────────────────────────────────────────────

def load_face_model(device: str):
    try:
        from insightface.app import FaceAnalysis
        ctx = 0 if device == "cuda" else -1
        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        app.prepare(ctx_id=ctx, det_size=(640, 640))
        print("  Loaded InsightFace buffalo_l (ArcFace)")
        return app
    except Exception as e:
        print(f"  InsightFace unavailable ({e}) — ID metric will be skipped")
        return None


def face_embedding(face_app, img: Image.Image) -> Optional[np.ndarray]:
    if face_app is None:
        return None
    arr = np.array(img.convert("RGB"))[:, :, ::-1]  # RGB→BGR
    faces = face_app.get(arr)
    if not faces:
        return None
    biggest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    e = biggest.normed_embedding.astype(np.float32)
    return e / np.linalg.norm(e)


# ─── File helpers ─────────────────────────────────────────────────────────────

def load_images(paths: List[Path]) -> List[Image.Image]:
    out = []
    for p in paths:
        try:
            out.append(Image.open(p).convert("RGB"))
        except Exception as ex:
            print(f"    WARN: cannot open {p}: {ex}")
    return out


def collect_slugs(base: Path, concept: str) -> Dict[str, List[Path]]:
    cd = base / concept
    if not cd.is_dir():
        return {}
    slugs: Dict[str, List[Path]] = {}
    for sd in sorted(cd.iterdir()):
        if not sd.is_dir():
            continue
        imgs = sorted(p for p in sd.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if imgs:
            slugs[sd.name] = imgs
    return slugs


def collect_ref_images(entry: dict, datasets_dir: Path, max_refs: int = 20) -> List[Path]:
    ds = Path(entry.get("dataset_dir", ""))
    if not ds.is_dir():
        ds = datasets_dir / entry.get("dataset", "")
    if not ds.is_dir():
        return []
    return sorted(p for p in ds.iterdir() if p.suffix.lower() in IMAGE_EXTS)[:max_refs]


# ─── Prompt templates ─────────────────────────────────────────────────────────

# OrthA Sec. 4: TA measures scene/style context only — no subject identity in text.
TA_CONTEXT_PERSON = {
    "portrait":     "neutral background, studio lighting",
    "cyberpunk":    "in the style of Cyberpunk 2077, 4K, ultra-realistic",
    "vangogh":      "in the style of Van Gogh, oil painting, expressive brush strokes",
    "forest":       "in a forest, natural daylight, bokeh background",
    "times_square": "in Times Square, street photography, neon lights",
    "superhero":    "wearing a superhero costume, dramatic lighting",
    "poker":        "playing poker, cinematic lighting, dramatic",
    "class_only":   "neutral background",
}
TA_CONTEXT_ANIMAL = {
    "portrait":    "close-up portrait, neutral background, studio lighting",
    "cyberpunk":   "in the style of Cyberpunk 2077, 4K, ultra-realistic, neon",
    "pixar":       "in the style of Pixar animation, 4K, colorful",
    "forest":      "in a forest, natural daylight, bokeh background",
    "beach":       "at the beach, sunny day, golden hour",
    "watercolor":  "watercolor style, soft colors",
    "class_only":  "neutral background",
}
TA_CONTEXT_OBJECT = {
    "portrait":    "product shot, white background, studio lighting",
    "cyberpunk":   "in the style of Cyberpunk 2077, 4K, ultra-realistic, neon",
    "watercolor":  "watercolor style, soft colors",
    "forest":      "placed in a forest, natural daylight",
    "fashion":     "worn by a model, fashion photography, editorial",
    "class_only":  "neutral background",
}

# Multi-concept scene → single slug for matched Table 2 baseline.
MATCHED_SINGLE_SLUG = {
    "plain": "portrait",
    "poker": "poker",
    "cyberpunk": "cyberpunk",
}

# Multi scene TA contexts (must match inference_cascade.SCENE_CONTEXTS).
MULTI_SCENE_TA = {
    "plain":     "group portrait, neutral background, studio lighting",
    "poker":     "playing poker together, cinematic lighting, dramatic",
    "cyberpunk": "in the style of Cyberpunk 2077, 4K, ultra-realistic, neon",
}


def build_generation_prompts(ctype: str, trigger: str, anchors: List[str]) -> Dict[str, str]:
    """Full inference prompts (with TI trigger) — used for TA per OrthA / Mix-of-Show."""
    anc = ", ".join(anchors) if anchors else ctype
    bare = (anchors[0] if anchors else ctype).removeprefix("a ").removeprefix("an ")

    person = {
        "portrait":     f"a photo of {trigger}, {anc}, neutral background, studio lighting",
        "cyberpunk":    f"a photo of {trigger}, in the style of Cyberpunk 2077, 4K, ultra-realistic",
        "vangogh":      f"a painting of {trigger}, in the style of Van Gogh, oil painting, expressive brush strokes",
        "forest":       f"a photo of {trigger}, in a forest, natural daylight, bokeh background",
        "times_square": f"a photo of {trigger}, in Times Square, street photography, neon lights",
        "superhero":    f"a photo of {trigger}, wearing a superhero costume, dramatic lighting",
        "poker":        f"a photo of {trigger}, playing poker, cinematic lighting, dramatic",
        "class_only":   f"a photo of a {bare}, neutral background",
    }
    animal = {
        "portrait":    f"a photo of {trigger}, close-up portrait, neutral background, studio lighting",
        "cyberpunk":   f"a photo of {trigger}, in the style of Cyberpunk 2077, 4K, ultra-realistic, neon",
        "pixar":       f"a photo of {trigger}, in the style of Pixar animation, 4K, colorful",
        "forest":      f"a photo of {trigger}, in a forest, natural daylight, bokeh background",
        "beach":       f"a photo of {trigger}, at the beach, sunny day, golden hour",
        "watercolor":  f"a painting of {trigger}, watercolor style, soft colors",
        "class_only":  f"a photo of a {bare}, neutral background",
    }
    obj = {
        "portrait":    f"a photo of {trigger}, product shot, white background, studio lighting",
        "cyberpunk":   f"a photo of {trigger}, in the style of Cyberpunk 2077, 4K, ultra-realistic, neon",
        "watercolor":  f"a painting of {trigger}, watercolor style, soft colors",
        "forest":      f"a photo of {trigger}, placed in a forest, natural daylight",
        "fashion":     f"a photo of {trigger}, worn by a model, fashion photography, editorial",
        "class_only":  f"a photo of a {bare}, neutral background",
    }
    if ctype in ("person", "man", "woman"):
        return person
    if ctype == "animal":
        return animal
    return obj


def build_ta_prompts(ctype: str) -> Dict[str, str]:
    """Style/context-only (diagnostic only — not used for paper TA)."""
    if ctype in ("person", "man", "woman"):
        return dict(TA_CONTEXT_PERSON)
    if ctype == "animal":
        return dict(TA_CONTEXT_ANIMAL)
    return dict(TA_CONTEXT_OBJECT)


def build_slug_prompts(ctype: str, trigger: str, anchors: List[str]) -> Dict[str, str]:
    """Full generation-style prompts (class word instead of OOV trigger) — for calibration."""
    cw = anchors[0] if anchors else ctype
    bare = cw.removeprefix("a ").removeprefix("an ")
    subject = cw

    person = {
        "portrait":     f"a photo of {subject}, {TA_CONTEXT_PERSON['portrait']}",
        "cyberpunk":    f"a photo of {subject}, {TA_CONTEXT_PERSON['cyberpunk']}",
        "vangogh":      f"a painting of {subject}, {TA_CONTEXT_PERSON['vangogh']}",
        "forest":       f"a photo of {subject}, {TA_CONTEXT_PERSON['forest']}",
        "times_square": f"a photo of {subject}, {TA_CONTEXT_PERSON['times_square']}",
        "superhero":    f"a photo of {subject}, {TA_CONTEXT_PERSON['superhero']}",
        "poker":        f"a photo of {subject}, {TA_CONTEXT_PERSON['poker']}",
        "class_only":   f"a photo of a {bare}, {TA_CONTEXT_PERSON['class_only']}",
    }
    animal = {
        "portrait":    f"a photo of {subject}, {TA_CONTEXT_ANIMAL['portrait']}",
        "cyberpunk":   f"a photo of {subject}, {TA_CONTEXT_ANIMAL['cyberpunk']}",
        "pixar":       f"a photo of {subject}, {TA_CONTEXT_ANIMAL['pixar']}",
        "forest":      f"a photo of {subject}, {TA_CONTEXT_ANIMAL['forest']}",
        "beach":       f"a photo of {subject}, {TA_CONTEXT_ANIMAL['beach']}",
        "watercolor":  f"a painting of {subject}, {TA_CONTEXT_ANIMAL['watercolor']}",
        "class_only":  f"a photo of a {bare}, {TA_CONTEXT_ANIMAL['class_only']}",
    }
    obj = {
        "portrait":    f"a photo of {subject}, {TA_CONTEXT_OBJECT['portrait']}",
        "cyberpunk":   f"a photo of {subject}, {TA_CONTEXT_OBJECT['cyberpunk']}",
        "watercolor":  f"a painting of {subject}, {TA_CONTEXT_OBJECT['watercolor']}",
        "forest":      f"a photo of {subject}, {TA_CONTEXT_OBJECT['forest']}",
        "fashion":     f"a photo of {subject}, {TA_CONTEXT_OBJECT['fashion']}",
        "class_only":  f"a photo of a {bare}, {TA_CONTEXT_OBJECT['class_only']}",
    }
    if ctype in ("person", "man", "woman"):
        return person
    if ctype == "animal":
        return animal
    return obj


def all_face_embeddings(face_app, img: Image.Image) -> List[np.ndarray]:
    """Normalized ArcFace embeddings for all detected faces."""
    if face_app is None:
        return []
    arr = np.array(img.convert("RGB"))[:, :, ::-1]
    faces = face_app.get(arr)
    out = []
    for f in faces:
        e = f.normed_embedding.astype(np.float32)
        out.append(e / np.linalg.norm(e))
    return out


def all_face_bboxes(face_app, img: Image.Image) -> List[Tuple[np.ndarray, tuple]]:
    """Return (embedding, bbox) for each detected face. bbox = (x1,y1,x2,y2)."""
    if face_app is None:
        return []
    arr = np.array(img.convert("RGB"))[:, :, ::-1]
    faces = face_app.get(arr)
    out = []
    for f in faces:
        e = f.normed_embedding.astype(np.float32)
        e = e / np.linalg.norm(e)
        bb = tuple(float(x) for x in f.bbox)
        out.append((e, bb))
    return out


def crop_face(img: Image.Image, bbox: tuple, margin: float = 0.15) -> Image.Image:
    """Crop face region with relative margin on bbox width/height."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    mx, my = w * margin, h * margin
    W, H = img.size
    left = max(0, int(x1 - mx))
    top = max(0, int(y1 - my))
    right = min(W, int(x2 + mx))
    bottom = min(H, int(y2 + my))
    if right <= left or bottom <= top:
        return img
    return img.crop((left, top, right, bottom))


def assign_faces_to_concepts(
    face_items: List[Tuple[np.ndarray, tuple]],
    ref_faces: Dict[str, List[np.ndarray]],
    concepts: List[str],
) -> Dict[str, Optional[Tuple[np.ndarray, tuple]]]:
    """Greedy assign each face to best-matching concept (one face per concept)."""
    assigned: Dict[str, Optional[Tuple[np.ndarray, tuple]]] = {c: None for c in concepts}
    if not face_items:
        return assigned

    pairs = []
    for fi, (emb, bbox) in enumerate(face_items):
        for concept in concepts:
            refs = ref_faces.get(concept, [])
            if not refs:
                continue
            score = max(float(np.dot(emb, rf)) for rf in refs)
            pairs.append((score, fi, concept))

    pairs.sort(reverse=True)
    used_faces: set = set()
    used_concepts: set = set()
    for score, fi, concept in pairs:
        if fi in used_faces or concept in used_concepts:
            continue
        if score <= ID_DETECT_COS_THRESHOLD:
            continue
        used_faces.add(fi)
        used_concepts.add(concept)
        assigned[concept] = face_items[fi]

    return assigned


def face_crop_ia(
    img: Image.Image,
    face_app,
    ref_clip: "torch.Tensor",
    concepts: List[str],
    ref_faces: Dict[str, List[np.ndarray]],
    backend, clip_model, clip_proc, device: str,
) -> Optional[float]:
    """IA on per-concept face crops (multi-composite protocol)."""
    items = all_face_bboxes(face_app, img)
    if not items:
        return None
    assignment = assign_faces_to_concepts(items, ref_faces, concepts)
    scores = []
    for concept in concepts:
        item = assignment.get(concept)
        if item is None or concept not in ref_clip:
            continue
        _, bbox = item
        crop = crop_face(img, bbox)
        feat = encode_images(backend, clip_model, clip_proc, [crop], device)
        scores.append(float((feat @ ref_clip[concept].T).mean().item()))
    return float(np.mean(scores)) if scores else None


# ─── Core metric computation ──────────────────────────────────────────────────

def compute_mode_metrics(
    slug_paths: Dict[str, List[Path]],
    gen_prompts: Dict[str, str],
    ref_paths: List[Path],
    backend, clip_model, clip_proc, clip_tok,
    face_app,
    device: str,
    is_person: bool,
) -> Dict:
    ref_images = load_images(ref_paths) if ref_paths else []

    # Pre-compute ref CLIP features
    ref_clip = None
    if ref_images:
        ref_clip = encode_images(backend, clip_model, clip_proc, ref_images, device)

    # Pre-compute ref face embeddings
    ref_faces: List[np.ndarray] = []
    if is_person and face_app is not None:
        for rim in ref_images:
            e = face_embedding(face_app, rim)
            if e is not None:
                ref_faces.append(e)

    per_slug: Dict[str, dict] = {}

    for slug, img_paths in slug_paths.items():
        gen_images = load_images(img_paths)
        if not gen_images:
            continue

        gen_clip = encode_images(backend, clip_model, clip_proc, gen_images, device)

        # ── Text Alignment ──
        ta: Optional[float] = None
        if slug not in SKIP_TA_SLUGS:
            prompt = gen_prompts.get(slug)
            if prompt:
                txt_feat = encode_texts(backend, clip_model, clip_proc, clip_tok,
                                        [prompt], device)
                cos = float((gen_clip @ txt_feat.T).mean().item())
                ta = clipscore_from_cosine(cos)

        # ── Image Alignment ──
        ia: Optional[float] = None
        if ref_clip is not None:
            ia = float((gen_clip @ ref_clip.T).mean().item())

        # ── Identity Alignment (ArcFace detection rate, paper-aligned) ──
        # Metric from OrthA supplement: fraction of generated images where
        # ArcFace distance to ANY reference face < 0.680 (cos_sim > 0.320).
        # Images with no detected face count as 0 (identity absent).
        id_rate: Optional[float] = None
        id_cos: Optional[float] = None   # raw cosine mean on detected faces (debug)
        if is_person and face_app is not None and ref_faces and slug not in SKIP_TA_SLUGS:
            detections = []
            detected_sims = []
            for im in gen_images:
                gfe = face_embedding(face_app, im)
                if gfe is None:
                    detections.append(0.0)
                else:
                    best_cos = float(max(np.dot(gfe, rfe) for rfe in ref_faces))
                    detected_sims.append(best_cos)
                    detections.append(1.0 if best_cos > ID_DETECT_COS_THRESHOLD else 0.0)
            id_rate = float(np.mean(detections))
            id_cos = float(np.mean(detected_sims)) if detected_sims else None

        per_slug[slug] = {
            "ta": ta, "ia": ia,
            "id": id_rate,    # detection rate 0..1 (paper metric)
            "id_cos": id_cos, # raw cosine mean on detected faces (debug)
            "n": len(gen_images),
        }

    def _avg(vals):
        v = [x for x in vals if x is not None]
        return round(float(np.mean(v)), 4) if v else None

    ta_list  = [v["ta"]     for s, v in per_slug.items() if s not in SKIP_TA_SLUGS]
    ia_list  = [v["ia"]     for v in per_slug.values()]
    id_list  = [v["id"]     for s, v in per_slug.items() if s not in SKIP_TA_SLUGS]
    cos_list = [v["id_cos"] for s, v in per_slug.items()
                if s not in SKIP_TA_SLUGS and v.get("id_cos") is not None]

    def _round(x):
        return round(x, 4) if isinstance(x, float) else x

    return {
        "per_slug": {s: {k: _round(val) for k, val in d.items()}
                     for s, d in per_slug.items()},
        "avg": {
            "ta":     _avg(ta_list),
            "ia":     _avg(ia_list),
            "id":     _avg(id_list),    # detection rate 0..1 (paper metric)
            "id_cos": _avg(cos_list),   # raw cosine on detected faces (debug)
        },
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Compute TA/IA/ID eval metrics")
    ap.add_argument("--eval_dir",      default="outputs/eval_infer")
    ap.add_argument("--registry",      default="concept_registry.json")
    ap.add_argument("--datasets_dir",  default="Datasets")
    ap.add_argument("--output",        default="outputs/eval_infer/metrics.json")
    ap.add_argument("--clip_model",    default="ViT-B-32",
                    help="CLIP backbone (Mix-of-Show / CLIPScore uses ViT-B-32)")
    ap.add_argument("--clip_pretrained", default="openai")
    ap.add_argument("--no_face",       action="store_true", help="Skip ArcFace ID metric")
    ap.add_argument("--device",        default="cuda")
    args = ap.parse_args()

    eval_dir   = Path(args.eval_dir)
    t2i_dir    = eval_dir / "t2i"
    i2i_dir    = eval_dir / "i2i"
    datasets_dir = Path(args.datasets_dir)

    if not Path(args.registry).exists():
        raise SystemExit(f"Registry not found: {args.registry}")
    registry = json.load(open(args.registry))

    print("Loading CLIP …")
    backend, clip_model, clip_proc, clip_tok = load_clip(
        args.clip_model, args.clip_pretrained, args.device
    )

    face_app = None
    if not args.no_face:
        print("Loading ArcFace …")
        face_app = load_face_model(args.device)

    results: Dict = {}

    for entry in tqdm(registry, desc="Concepts"):
        concept = entry.get("concept")
        if not concept or entry.get("type", "unknown") == "unknown":
            continue

        ctype   = entry.get("type", "person")
        trigger = entry.get("trigger") or f"<id_{concept}>"
        anchors = entry.get("attribute_anchors") or []
        is_person = ctype in ("person", "man", "woman")

        ref_paths   = collect_ref_images(entry, datasets_dir)
        gen_prompts = build_generation_prompts(ctype, trigger, anchors)

        concept_res: Dict = {"type": ctype, "trigger": trigger}

        for mode_name, mode_dir in [("t2i", t2i_dir), ("i2i", i2i_dir)]:
            if not mode_dir.is_dir():
                continue
            slugs = collect_slugs(mode_dir, concept)
            if not slugs:
                continue
            tqdm.write(f"  {concept:16s} [{mode_name}]  {len(slugs)} slugs")
            concept_res[mode_name] = compute_mode_metrics(
                slugs, gen_prompts, ref_paths,
                backend, clip_model, clip_proc, clip_tok,
                face_app, args.device, is_person,
            )

        if len(concept_res) > 2:
            results[concept] = concept_res

    # ── Save JSON ──
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # ── Print table ──
    header = (f"{'Concept':<16} {'Type':<8} {'Mode':<5}"
              f" {'TA':>7} {'IA':>7} {'ID@.68':>7} {'ID_cos':>7}")
    print("\n" + "=" * len(header))
    print(header)
    print(f"  TA=CLIPScore max(0,cos)*{CLIPSCORE_W} w/ generation prompt  IA=CLIP image-image  ID@.68=ArcFace")
    print("-" * len(header))
    for concept in sorted(results):
        cr    = results[concept]
        ctype = cr.get("type", "?")
        for mode in ("t2i", "i2i"):
            if mode not in cr:
                continue
            avg  = cr[mode].get("avg", {})
            ta   = f"{avg['ta']:.4f}"     if avg.get("ta")     is not None else "   —  "
            ia   = f"{avg['ia']:.4f}"     if avg.get("ia")     is not None else "   —  "
            id_  = f"{avg['id']:.4f}"     if avg.get("id")     is not None else "   —  "
            cos_ = f"{avg['id_cos']:.4f}" if avg.get("id_cos") is not None else "   —  "
            print(f"{concept:<16} {ctype:<8} {mode:<5} {ta:>7} {ia:>7} {id_:>7} {cos_:>7}")
    print("=" * len(header))

    # ── Save CSV ──
    csv_path = out_path.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["concept", "type", "mode", "TA", "IA", "ID_rate_at_0.68", "ID_cos_debug"])
        for concept in sorted(results):
            cr = results[concept]
            for mode in ("t2i", "i2i"):
                if mode not in cr:
                    continue
                avg = cr[mode].get("avg", {})
                w.writerow([concept, cr["type"], mode,
                             avg.get("ta", ""), avg.get("ia", ""),
                             avg.get("id", ""), avg.get("id_cos", "")])
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
