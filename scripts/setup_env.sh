#!/bin/bash
# ============================================================
# setup_env.sh — install all Python deps for LILAC
# (Qwen-Image-Edit LoRA training + TI + eval + ablation).
#
# Usage:
#   bash scripts/setup_env.sh                         # auto-detect venv, default cu128
#   bash scripts/setup_env.sh /path/to/venv           # create / use given venv path
#   bash scripts/setup_env.sh --no-venv               # install into current Python
#
# Environment knobs:
#   PYTHON_BIN=python3.11   override Python interpreter used to create the venv
#   CUDA_VERSION=cu128      torch wheel index suffix (cu121, cu124, cu128, cpu)
#   FORCE_REINSTALL=1       pip install --force-reinstall (slow, fixes broken envs)
#   SKIP_TORCH=1            skip the torch install step (when you trust your env)
#   SKIP_BNB=1              skip bitsandbytes (CPU-only machines)
#   DRY_RUN=1               print what would be installed, no changes
#
# Notes:
#   - Works both on fresh machines (creates venv) AND inside an already-active
#     venv (just installs / upgrades the required packages).
# ============================================================

set -e

VENV_ARG=${1:-}
PYTHON_BIN=${PYTHON_BIN:-python3}
CUDA_VERSION=${CUDA_VERSION:-cu128}
FORCE_REINSTALL=${FORCE_REINSTALL:-0}
SKIP_TORCH=${SKIP_TORCH:-0}
SKIP_BNB=${SKIP_BNB:-0}
DRY_RUN=${DRY_RUN:-0}

# ---- Resolve venv strategy -----------------------------------------------
case "${VENV_ARG}" in
    --no-venv)
        USE_VENV=0
        VENV_PATH=""
        ;;
    "")
        # No arg: use $VIRTUAL_ENV if already active, else default ./venv
        if [ -n "${VIRTUAL_ENV:-}" ]; then
            USE_VENV=1
            VENV_PATH="${VIRTUAL_ENV}"
            VENV_EXISTING=1
        else
            USE_VENV=1
            VENV_PATH="./venv"
        fi
        ;;
    *)
        USE_VENV=1
        VENV_PATH="${VENV_ARG}"
        ;;
esac

# ---- Create + activate venv (if requested) -------------------------------
if [ "${USE_VENV}" = "1" ]; then
    if [ ! -d "${VENV_PATH}" ]; then
        echo "Creating venv at ${VENV_PATH} (interpreter: ${PYTHON_BIN})..."
        [ "${DRY_RUN}" = "1" ] || ${PYTHON_BIN} -m venv "${VENV_PATH}"
    else
        echo "Re-using existing venv at ${VENV_PATH}"
    fi
    # shellcheck disable=SC1091
    [ "${DRY_RUN}" = "1" ] || source "${VENV_PATH}/bin/activate"
fi

PIP_FLAGS=""
[ "${FORCE_REINSTALL}" = "1" ] && PIP_FLAGS="--force-reinstall --no-deps"

run_pip() {
    if [ "${DRY_RUN}" = "1" ]; then
        echo "  [DRY] pip install $*"
    else
        pip install ${PIP_FLAGS} "$@"
    fi
}

echo ""
echo "============================================"
echo " LILAC env setup"
echo "   venv         : $([ "${USE_VENV}" = "1" ] && echo "${VENV_PATH}" || echo "(current Python, --no-venv)")"
echo "   python       : $(which ${PYTHON_BIN} 2>/dev/null || echo '?')"
echo "   torch wheels : https://download.pytorch.org/whl/${CUDA_VERSION}"
echo "   force        : $([ "${FORCE_REINSTALL}" = "1" ] && echo yes || echo no)"
echo "   dry_run      : $([ "${DRY_RUN}" = "1" ] && echo yes || echo no)"
echo "============================================"
echo ""

# ---- 1. Upgrade installer toolchain --------------------------------------
echo "[1/7] Upgrading pip + wheel + setuptools"
run_pip --upgrade pip wheel setuptools

# ---- 2. PyTorch with CUDA support ----------------------------------------
if [ "${SKIP_TORCH}" = "1" ]; then
    echo "[2/7] Skipping torch (SKIP_TORCH=1). Make sure your env has torch >= 2.4 + CUDA."
else
    echo "[2/7] Installing torch + torchvision (${CUDA_VERSION})"
    if [ "${CUDA_VERSION}" = "cpu" ]; then
        run_pip torch torchvision
    else
        run_pip torch torchvision \
            --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"
    fi
fi

# ---- 3. Diffusers stack (Qwen-Image + Qwen-Image-Edit) -------------------
echo "[3/7] diffusers + transformers + accelerate"
# diffusers >= 0.38 ships QwenImagePipeline + QwenImageEditPipeline.
# transformers >= 4.49 registers Qwen2_5_VLForConditionalGeneration (required
# by both Qwen-Image and Qwen-Image-Edit as their text encoder backbone).
# Lower versions throw "cannot import name 'Qwen2_5_VLForConditionalGeneration'".
run_pip "diffusers>=0.38" "transformers>=4.49" "accelerate>=0.30"

# ---- 4. PEFT (LoRA + DoRA) -----------------------------------------------
echo "[4/7] peft (LoRA + DoRA + magnitude vector)"
# DoRA requires peft >= 0.10.
run_pip "peft>=0.10"

# ---- 5. 8-bit Adam (memory-saver during LoRA training) -------------------
if [ "${SKIP_BNB}" = "1" ]; then
    echo "[5/7] Skipping bitsandbytes (SKIP_BNB=1). Trainer will fall back to AdamW."
else
    echo "[5/7] bitsandbytes (for --use_8bit_adam)"
    run_pip bitsandbytes
fi

# ---- 6. Eval metrics + plotting ------------------------------------------
echo "[6/7] open_clip_torch + pandas + matplotlib"
run_pip open_clip_torch pandas matplotlib

# ---- 7. Misc utilities ---------------------------------------------------
echo "[7/8] safetensors + Pillow + tqdm + numpy"
run_pip safetensors Pillow tqdm numpy

# ---- 8. Qwen-VL utils + InsightFace (ArcFace ID metric) -----------------
echo "[8/8] qwen-vl-utils + insightface + onnxruntime-gpu"
run_pip qwen-vl-utils
# insightface needs C++ headers; on most A100 nodes these are pre-installed
run_pip insightface onnxruntime-gpu || {
    echo "  [WARN] onnxruntime-gpu failed — falling back to CPU onnxruntime"
    run_pip insightface onnxruntime
}

# ---- Verification --------------------------------------------------------
if [ "${DRY_RUN}" = "1" ]; then
    echo ""
    echo "DRY_RUN=1: skipping verification."
    exit 0
fi

echo ""
echo "============================================"
echo " Verification"
echo "============================================"
python <<'PY'
import sys
print(f"  python      : {sys.version.split()[0]}")

import torch
print(f"  torch       : {torch.__version__}")
print(f"  CUDA avail  : {torch.cuda.is_available()}")
print(f"  CUDA version: {torch.version.cuda}")
print(f"  GPU count   : {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        print(f"    GPU {i}: {torch.cuda.get_device_name(i)}  ({total / 1e9:.1f} GB)")

import diffusers
print(f"  diffusers   : {diffusers.__version__}")
try:
    from diffusers import QwenImagePipeline, QwenImageEditPipeline  # noqa: F401
    print("    -> QwenImagePipeline + QwenImageEditPipeline importable")
except ImportError as e:
    print(f"    !! Qwen pipelines NOT importable: {e}")
    print("    !! upgrade with: pip install --upgrade diffusers>=0.38")

import transformers
print(f"  transformers: {transformers.__version__}")
try:
    from transformers import Qwen2_5_VLForConditionalGeneration  # noqa: F401
    print("    -> Qwen2_5_VLForConditionalGeneration importable")
except ImportError:
    print("    !! Qwen2_5_VLForConditionalGeneration NOT importable")
    print("    !! upgrade with: pip install --upgrade 'transformers>=4.49'")

import peft
print(f"  peft        : {peft.__version__}")
try:
    from peft import LoraConfig
    LoraConfig(r=4, lora_alpha=4, use_dora=True)
    print("    -> DoRA (use_dora=True) supported")
except TypeError:
    print("    !! DoRA NOT supported, upgrade peft >= 0.10")

import accelerate
print(f"  accelerate  : {accelerate.__version__}")

try:
    import bitsandbytes as bnb
    print(f"  bitsandbytes: {bnb.__version__}")
except ImportError:
    print("  bitsandbytes: NOT installed (8bit Adam will fall back to torch.optim.AdamW)")

try:
    import open_clip
    print(f"  open_clip   : {open_clip.__version__}")
except ImportError:
    print("  open_clip   : NOT installed (eval will use transformers CLIP fallback)")

try:
    import qwen_vl_utils  # noqa: F401
    print("  qwen_vl_utils: OK")
except ImportError:
    print("  qwen_vl_utils: NOT installed  (pip install qwen-vl-utils)")

try:
    import insightface  # noqa: F401
    print(f"  insightface : {insightface.__version__}")
except ImportError:
    print("  insightface : NOT installed  (ID metric will be skipped)")

print()
print("  scripts compile check:")
import ast, glob, os
errs = 0
for f in sorted(glob.glob("scripts/*.py")):
    try:
        ast.parse(open(f).read())
        print(f"    OK  {f}")
    except SyntaxError as e:
        print(f"    !!  {f}: {e}")
        errs += 1
sys.exit(errs)
PY

echo ""
echo "============================================"
echo " Done."
if [ "${USE_VENV}" = "1" ]; then
    echo "   To use this env in a new shell:"
    echo "     source ${VENV_PATH}/bin/activate"
fi
echo ""
echo "   Quick sanity test (downloads model the first time):"
echo "     python -c \"from diffusers import QwenImageEditPipeline; print('ok')\""
echo ""
echo "   Train all concepts (after Datasets/ + concept_registry.json are in place):"
echo "     bash scripts/run_train_all.sh"
echo "============================================"
