#!/usr/bin/env python3
"""M0 dependency gate — faithful Vec2Text (pretrained gtr-base) smoke test.

Loads the REAL vec2text dependency (bind-mounted .deps/), GTR-base encoder, and the
pretrained gtr-base corrector; inverts 4 clean texts (20 steps + sequence beam). Pass =
it imports + inverts on the AMD iGPU. This is the highest-risk gate (vec2text 2023 code
vs the container's transformers 5.x)."""
import sys, os
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, ".deps"))
# transformers 4.44's T5 uses apex FusedRMSNorm (JIT-compiled); needs a writable
# TORCH_EXTENSIONS_DIR (set in the launch env to /tmp, the container default ~/.cache
# is read-only here).
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
import vec2text
print("vec2text imported OK", flush=True)
from transformers import AutoModel, AutoTokenizer

dev = "cuda" if torch.cuda.is_available() else "cpu"
enc = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to(dev).eval()
tokz = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")


def get_gtr(texts):
    inp = tokz(texts, return_tensors="pt", max_length=128, truncation=True, padding="max_length").to(dev)
    with torch.no_grad():
        out = enc(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
        emb = vec2text.models.model_utils.mean_pool(out.last_hidden_state, inp["attention_mask"])
    return emb


texts = [
    "Jack Morris is a researcher at Cornell Tech working on embeddings.",
    "The Eiffel Tower is located in Paris, the capital of France.",
    "Differential privacy protects data by adding calibrated noise.",
    "Vec2Text iteratively inverts dense text embeddings back to text.",
]
emb = get_gtr(texts)
print("gtr emb", tuple(emb.shape), flush=True)
cor = vec2text.load_pretrained_corrector("gtr-base")
print("corrector loaded OK", flush=True)
rec = vec2text.invert_embeddings(embeddings=emb.to(dev), corrector=cor, num_steps=20, sequence_beam_width=4)
print("=== inversion ===", flush=True)
for t, r in zip(texts, rec):
    print("TRUE:", t)
    print("REC :", r)
    print(flush=True)
print("[M0] PASS — vec2text imports + inverts on this device", flush=True)
