"""Clean Vec2Text embedding-inversion attack handler (Morris et al. 2023).

Faithful pretrained attack on a POOLED sentence embedding — the surface where
Vec2Text is sound (a single bottleneck vector compressing the whole sequence),
unlike the per-position residual stream where its iterative feedback is moot
(see ``docs/dev/sae-attack.md`` §Vec2Text and the dp-stronger-attacks B7/B8 logs).

Encoder φ = ``sentence-transformers/gtr-t5-base`` (a real RAG/retrieval sentence
embedder), mean-pooled to a 768-d vector. The attacker inverts the *released*
embedding back to text with vec2text's PRETRAINED ``gtr-base`` corrector
(iterative re-embed + sequence beam). A differential-privacy defense is an
*external* transform on the released embedding (``dp_noise``) — scheme-agnostic,
the handler asserts nothing about any defense.

Threat model (WEIGHTS-PUB analog for embeddings): the attacker has the public
encoder φ, the published DP params (clip C, σ), and the corrector matched to φ.

Heavy dependency note: ``vec2text`` + a ``transformers==4.44`` shadow live in the
bind-mounted ``.deps/`` (kept OUT of the core ``talens`` library). This module
bootstraps that path and points apex's JIT cache at a writable ``/tmp`` dir, so it
must run inside the ROCm container (``scripts/run_in_rocm.sh``). Recipe + rationale:
memory ``vec2text-rocm-dependency-recipe``.
"""
from __future__ import annotations

import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
# vec2text + transformers-4.44 shadow (transformers 5.x rejects vec2text's nested
# from_pretrained under its meta-device init); torch stays the container's ROCm build.
sys.path.insert(0, str(_REPO / ".deps"))
# transformers 4.44's T5 uses apex FusedRMSNorm, JIT-compiled into TORCH_EXTENSIONS_DIR;
# the container's ~/.cache is read-only, so point HOME + the ext dir at writable /tmp.
os.environ.setdefault("HOME", "/tmp/v2thome")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_ext")
for _d in (os.environ["HOME"], os.environ["TORCH_EXTENSIONS_DIR"]):
    os.makedirs(_d, exist_ok=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import vec2text  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

GTR_MODEL = "sentence-transformers/gtr-t5-base"
_WS = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    return _WS.sub(" ", s).strip().lower()


def token_f1(pred: str, true: str) -> float:
    """Word-multiset F1 of a reconstruction against the source text."""
    p, t = normalize_text(pred).split(), normalize_text(true).split()
    if not p or not t:
        return 0.0
    overlap = sum((Counter(p) & Counter(t)).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(t)
    return 2 * prec * rec / (prec + rec)


def gaussian_sigma(clip_C: float, epsilon: float, delta: float = 1e-5) -> float:
    """(ε,δ)-Gaussian-mechanism σ for an L2-sensitivity == clip norm C.

    ``epsilon == inf`` → 0 (no noise, clip only)."""
    if math.isinf(epsilon):
        return 0.0
    z = math.sqrt(2 * math.log(1.25 / delta))
    return clip_C * z / epsilon


def dp_noise(emb: np.ndarray, *, clip_C: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """External DP defense on the released embedding: clip each row to norm
    ``clip_C`` then add isotropic Gaussian ``N(0, σ²)``. ``sigma == 0`` → clip only."""
    scale = np.minimum(1.0, clip_C / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9))
    out = (emb * scale).astype(np.float32)
    if sigma > 0:
        out = out + sigma * rng.standard_normal(out.shape).astype(np.float32)
    return out


class Vec2TextAttack:
    """Pretrained Vec2Text inversion of pooled GTR sentence embeddings."""

    def __init__(self, device: str | None = None, corrector: str = "gtr-base"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = AutoModel.from_pretrained(GTR_MODEL).encoder.to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(GTR_MODEL)
        self.corrector = vec2text.load_pretrained_corrector(corrector)

    def canonicalize(self, texts: list[str], max_tokens: int = 32, min_tokens: int = 8) -> list[str]:
        """Truncate texts to ``max_tokens`` GTR tokens and round-trip through the
        tokenizer, so the recovery ground-truth is exactly what φ encodes (Morris
        32-token regime). Drops texts shorter than ``min_tokens``."""
        out = []
        for t in texts:
            ids = self.tokenizer(t, truncation=True, max_length=max_tokens)["input_ids"]
            if len(ids) >= min_tokens:
                out.append(self.tokenizer.decode(ids, skip_special_tokens=True))
        return out

    @torch.no_grad()
    def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """φ: texts → mean-pooled 768-d GTR embeddings (n, 768)."""
        chunks = []
        for i in range(0, len(texts), batch_size):
            inp = self.tokenizer(texts[i:i + batch_size], return_tensors="pt", max_length=128,
                                 truncation=True, padding="max_length").to(self.device)
            o = self.encoder(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
            pooled = vec2text.models.model_utils.mean_pool(o.last_hidden_state, inp["attention_mask"])
            chunks.append(pooled.cpu())
        return torch.cat(chunks, 0).numpy().astype(np.float32)

    def invert(self, emb: np.ndarray, num_steps: int = 20, beam: int = 1) -> list[str]:
        """Invert embeddings → text. ``num_steps == 0`` runs the base inversion
        model (no correction); else iterative correction with ``beam`` sequence beams."""
        et = torch.from_numpy(np.asarray(emb, dtype=np.float32)).to(self.device)
        return vec2text.invert_embeddings(
            embeddings=et, corrector=self.corrector,
            num_steps=(None if num_steps == 0 else num_steps),
            sequence_beam_width=(0 if num_steps == 0 else beam),
        )

    def score(self, recon: list[str], true: list[str], clean_emb: np.ndarray | None = None) -> dict:
        """Recovery vs the source text: BLEU, token-F1, exact-match, and (if
        ``clean_emb`` given) cos(φ(recon), e0)."""
        import sacrebleu
        bleu = float(np.mean([sacrebleu.sentence_bleu(r, [t]).score for r, t in zip(recon, true)]))
        tf1 = float(np.mean([token_f1(r, t) for r, t in zip(recon, true)]))
        exact = float(np.mean([normalize_text(r) == normalize_text(t) for r, t in zip(recon, true)]))
        out = {"bleu": bleu, "token_f1": tf1, "exact": exact}
        if clean_emb is not None:
            er = self.embed(recon)
            out["cos"] = float(np.mean((er * clean_emb).sum(1) /
                                       (np.linalg.norm(er, axis=1) * np.linalg.norm(clean_emb, axis=1) + 1e-9)))
        return out
