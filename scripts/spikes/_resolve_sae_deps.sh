#!/usr/bin/env bash
# Resolve sae-lens deps INSIDE the ROCm container WITHOUT clobbering the
# inherited ROCm torch/torchvision. sae-lens + transformer-lens are
# installed --no-deps (they are what drag in a PyPI torchvision that
# exact-pins and downgrades torch); everything else installs normally
# (no torch upper-pin), so ROCm torch 2.10 + torchvision 0.25 survive.
#
# Used both for the throwaway probe and as the Containerfile recipe.
set -euo pipefail

pip install --no-cache-dir --no-deps sae-lens transformer-lens
pip install --no-cache-dir \
    datasets accelerate pandas rich sentencepiece fancy-einsum \
    better-abc beartype jaxtyping typeguard pydantic simple-parsing \
    einops wandb

# Auto-resolve any remaining pure-python imports transformer_lens/sae_lens
# pull at import time (install --no-deps so none can touch torch).
for _ in $(seq 1 12); do
    missing=$(python3 -c 'import torch; from sae_lens import SAE' 2>&1 \
        | sed -nE "s/.*No module named '([^']+)'.*/\1/p" | head -1)
    if [ -z "$missing" ]; then
        python3 - <<'PY'
import torch, torchvision
from sae_lens import SAE
print("=== IMPORT OK ===")
print("torch     ", torch.__version__)
print("torchvision", torchvision.__version__)
print("cuda_avail", torch.cuda.is_available())
PY
        exit 0
    fi
    echo "missing: [$missing] -> installing --no-deps"
    pip install --no-cache-dir --no-deps "$missing"
done
echo "ERROR: unresolved imports after 12 rounds" >&2
python3 -c 'import torch; from sae_lens import SAE' || true
exit 1
