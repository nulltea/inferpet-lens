# Split-and-Denoise (SnD) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Split-and-Denoise (SnD) defense — dχ-privacy on token embeddings + a noise-aware transformer denoiser — and an eval that measures utility recovery of the pooled output embedding across the privacy budget η.

**Architecture:** A `DxPrivacy` forward-hook (sibling to `LocalDP`) privatizes input token embeddings; a `Denoiser(nn.Module)` reconstructs the clean pooled output embedding from `(e_n, X̃, Z)`; `snd_utility_sweep.py` orchestrates capture + denoiser train/infer over an η sweep, reporting cos/MSE recovery + perplexity degradation.

**Tech Stack:** Python, PyTorch, HuggingFace transformers, numpy, pytest. ROCm container for GPU runs (`scripts/run_in_rocm.sh`); host `.venv` (CPU torch) for model-free unit tests.

## Global Constraints

- Defenses live in `scripts/defenses/`; runnable evals in `scripts/evals/` — never logic in the core.
- η is the dχ budget; it is **not** the Gaussian (ε,δ) of `LocalDP`. Flag this in output JSON.
- Clip bound `C` = high-percentile token-norm (default 99.9), same convention as `dp_leakage_sweep.py`.
- fp32, `attn_implementation="eager"`, `pad_token=eos`, right padding (batch-invariance).
- One GPU process at a time; ROCm container for anything touching the model. Host `.venv` is CPU-only.
- Pooled embedding `e_*` = mean of last-hidden over real (non-pad) tokens.
- `e_c/e_n/e_d` ∈ R^d (d = model hidden = 768 for pythia-160m); `X̃, Z` ∈ R^{T×d}.

---

### Task 1: dχ-privacy mechanism (`DxPrivacy`)

**Files:**
- Create: `scripts/defenses/snd.py`
- Test: `tests/test_snd.py`

**Interfaces:**
- Produces: `DxPrivacy(C: float, eta: float)` — callable forward hook `(mod, inp, out) -> Tensor`. `eta=inf` ⇒ clip-only (no noise). Output same shape/dtype as `out`; every row L2 ≤ C.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_snd.py
import math
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from defenses.snd import DxPrivacy


def _apply(hook, x):
    """Invoke the forward hook the way register_forward_hook would: out is the embedding tensor."""
    return hook(None, None, x)


def test_dxprivacy_clips_to_C():
    torch.manual_seed(0)
    x = torch.randn(4, 7, 16) * 5.0          # (B, T, d), large norms
    out = _apply(DxPrivacy(C=2.0, eta=10.0), x)
    assert out.shape == x.shape
    norms = out.float().norm(dim=-1)
    assert torch.all(norms <= 2.0 + 1e-4), norms.max().item()


def test_dxprivacy_noise_scales_with_eta():
    torch.manual_seed(0)
    x = torch.randn(8, 12, 16)
    # large C so clipping rarely binds → Z reflects the raw mechanism noise
    z_small_eta = (_apply(DxPrivacy(C=1e6, eta=1.0), x) - x).norm(dim=-1).mean()
    z_large_eta = (_apply(DxPrivacy(C=1e6, eta=50.0), x) - x).norm(dim=-1).mean()
    assert z_small_eta > z_large_eta            # smaller η = more noise


def test_dxprivacy_inf_eta_is_clip_only():
    x = torch.randn(3, 5, 16) * 0.1             # norms < C → clip is a no-op
    out = _apply(DxPrivacy(C=100.0, eta=math.inf), x)
    assert torch.allclose(out, x)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_snd.py -k dxprivacy -v`
Expected: FAIL (`ModuleNotFoundError: defenses.snd` / `DxPrivacy` undefined)

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/defenses/snd.py
"""Split-and-Denoise (SnD) defense — dχ-privacy on token embeddings + a noise-aware denoiser.

Mai et al. 2023, arXiv:2310.09130 (paper:mai2023_splitanddenoise_protect_large).
Reference impl: https://github.com/NusIoraPrivacy/eaas-privacy

Repo rule: defenses live in scripts/defenses/. evals import and apply them.
DxPrivacy is a forward hook on the input-embedding layer (sibling to LocalDP), but uses
the dχ-privacy Laplacian (Wu et al. 2017) instead of the Gaussian mechanism:
  direction v = g/||g||, g~N(0,I_d); magnitude l~Gamma(d, 1/eta); M(x)=x+l*v; clip to C.
η is the dχ budget (larger = weaker privacy); it is NOT the Gaussian (ε,δ).
"""
from __future__ import annotations

import math
import torch


class DxPrivacy:
    """dχ-privacy mechanism as a forward hook on the token-embedding layer.

    eta=inf ⇒ clip-only (≈ clean). Per-row: add d-dim Laplacian noise, then L2-clip to C.
    """

    def __init__(self, C: float, eta: float):
        self.C, self.eta = C, eta

    def __call__(self, mod, inp, out):  # noqa: ARG002 (hook signature)
        f = out.float()
        if math.isfinite(self.eta):
            g = torch.randn_like(f)
            v = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-9)        # unit sphere
            # l ~ Gamma(shape=d, scale=1/eta), one magnitude per row
            d = f.shape[-1]
            gamma = torch.distributions.Gamma(torch.tensor(float(d)), torch.tensor(float(self.eta)))
            l = gamma.sample(f.shape[:-1]).to(f).unsqueeze(-1)          # (..., 1)
            f = f + l * v
        n = f.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        f = f * (self.C / n).clamp_max(1.0)                            # clip to C
        return f.to(out.dtype)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_snd.py -k dxprivacy -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/defenses/snd.py tests/test_snd.py
git commit -m "feat(defenses): SnD dχ-privacy mechanism (DxPrivacy hook)"
```

---

### Task 2: noise-aware Denoiser

**Files:**
- Modify: `scripts/defenses/snd.py`
- Test: `tests/test_snd.py`

**Interfaces:**
- Consumes: nothing from Task 1 (independent module in the same file).
- Produces: `Denoiser(d: int, n_layers: int = 3, n_heads: int = 8)` — `nn.Module`. `forward(e_n, X_tilde, Z, pad_mask=None) -> e_d` where `e_n: (B, d)`, `X_tilde/Z: (B, T, d)`, `pad_mask: (B, T)` bool (True = pad), returns `(B, d)`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_snd.py
from defenses.snd import Denoiser


def test_denoiser_output_shape():
    torch.manual_seed(0)
    B, T, d = 2, 5, 16
    m = Denoiser(d=d, n_layers=2, n_heads=4)
    e_d = m(torch.randn(B, d), torch.randn(B, T, d), torch.randn(B, T, d))
    assert e_d.shape == (B, d)


def test_denoiser_overfits_toward_clean():
    """A few steps on a fixed batch should pull denoised cos-to-e_c above the raw cos(e_n,e_c)."""
    torch.manual_seed(0)
    B, T, d = 8, 4, 16
    e_c = torch.randn(B, d)
    Z = 0.5 * torch.randn(B, T, d)
    X = torch.randn(B, T, d)
    e_n = e_c + 0.8 * torch.randn(B, d)                 # noised observation of e_c
    cos = torch.nn.functional.cosine_similarity
    base = cos(e_n, e_c).mean().item()
    m = Denoiser(d=d, n_layers=2, n_heads=4)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(150):
        opt.zero_grad()
        loss = ((m(e_n, X, Z) - e_c) ** 2).mean()
        loss.backward(); opt.step()
    after = cos(m(e_n, X, Z), e_c).mean().item()
    assert after > base, (base, after)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_snd.py -k denoiser -v`
Expected: FAIL (`Denoiser` undefined)

- [ ] **Step 3: Write minimal implementation**

```python
# append to scripts/defenses/snd.py
import torch.nn as nn


class Denoiser(nn.Module):
    """Noise-aware transformer: reconstruct clean pooled embedding e_c from (e_n, X̃, Z).

    Token sequence [e_n] ++ X̃ ++ Z (length 2T+1), + a learned type embedding (out/raw/noise)
    + sinusoidal position, L encoder layers; read the e_n slot → linear → e_d. One model serves
    the whole η sweep (conditions on Z). Padding masked via src_key_padding_mask.
    """

    def __init__(self, d: int, n_layers: int = 3, n_heads: int = 8):
        super().__init__()
        self.d = d
        self.type_emb = nn.Embedding(3, d)              # 0=output, 1=raw, 2=noise
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=d,
            activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d, d)

    @staticmethod
    def _posenc(T: int, d: int, device) -> torch.Tensor:
        pos = torch.arange(T, device=device).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2, device=device).float() * (-math.log(10000.0) / d))
        pe = torch.zeros(T, d, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        return pe

    def forward(self, e_n, X_tilde, Z, pad_mask=None):
        B, T, d = X_tilde.shape
        seq = torch.cat([e_n.unsqueeze(1), X_tilde, Z], dim=1)          # (B, 2T+1, d)
        types = torch.cat([
            torch.zeros(1, dtype=torch.long, device=seq.device),
            torch.ones(T, dtype=torch.long, device=seq.device),
            torch.full((T,), 2, dtype=torch.long, device=seq.device),
        ])
        seq = seq + self.type_emb(types).unsqueeze(0)
        pe = self._posenc(T, d, seq.device)
        seq = seq + torch.cat([torch.zeros(1, d, device=seq.device), pe, pe], dim=0).unsqueeze(0)
        kpm = None
        if pad_mask is not None:                                       # (B, T) → (B, 2T+1)
            f = torch.zeros(B, 1, dtype=torch.bool, device=seq.device)
            kpm = torch.cat([f, pad_mask, pad_mask], dim=1)
        h = self.enc(seq, src_key_padding_mask=kpm)
        return self.head(h[:, 0])                                       # e_n slot → e_d
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_snd.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/defenses/snd.py tests/test_snd.py
git commit -m "feat(defenses): SnD noise-aware transformer Denoiser"
```

---

### Task 3: utility-recovery metric helpers

**Files:**
- Create: `scripts/evals/snd_utility_sweep.py` (helpers first; main in Task 4)
- Test: `tests/test_snd.py`

**Interfaces:**
- Produces: `recovery_metrics(e_c, e_n, e_d) -> dict` with keys `cos_noised, cos_denoised, mse_noised, mse_denoised, recovery_cos, recovery_mse`. All inputs `np.ndarray (N, d)`. `recovery_cos = (cos_d − cos_n)/(1 − cos_n)`; `recovery_mse = 1 − mse_d/mse_n`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_snd.py
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "evals"))
from snd_utility_sweep import recovery_metrics


def test_recovery_metrics_perfect_denoise():
    rng = np.random.default_rng(0)
    e_c = rng.standard_normal((10, 8)).astype(np.float32)
    e_n = e_c + rng.standard_normal((10, 8)).astype(np.float32)   # noised
    e_d = e_c.copy()                                             # perfect recovery
    m = recovery_metrics(e_c, e_n, e_d)
    assert m["recovery_cos"] > 0.99 and m["recovery_mse"] > 0.99
    assert m["cos_denoised"] > m["cos_noised"]


def test_recovery_metrics_no_denoise():
    rng = np.random.default_rng(1)
    e_c = rng.standard_normal((10, 8)).astype(np.float32)
    e_n = e_c + rng.standard_normal((10, 8)).astype(np.float32)
    m = recovery_metrics(e_c, e_n, e_n.copy())                   # e_d == e_n
    assert abs(m["recovery_cos"]) < 1e-5 and abs(m["recovery_mse"]) < 1e-5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_snd.py -k recovery -v`
Expected: FAIL (`snd_utility_sweep` undefined)

- [ ] **Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
"""SnD utility-recovery sweep — the UTILITY axis of the SnD privacy–utility tradeoff.

Privatize token embeddings with dχ-privacy (DxPrivacy), let the rest of the model produce the
pooled output embedding e_n, denoise it locally with a noise-aware transformer (Denoiser) →
e_d, and measure how much of the clean pooled embedding e_c the denoiser recovers across the
budget η. Also report teacher-forced perplexity/acc degradation under the dχ noise (generation-
utility cost; the denoiser does not touch that surface). Privacy axis stays in dp_leakage_sweep.

η is the dχ budget, NOT the Gaussian ε of dp_leakage_sweep — not interchangeable.

GPU: ONE process at a time; run via scripts/run_in_rocm.sh. Output JSON under refine-logs/snd/.

  scripts/run_in_rocm.sh python3 scripts/evals/snd_utility_sweep.py \
      --etas inf,100,50,10,1 --train-etas 50,10,1 --out refine-logs/snd/snd_utility_sweep.json
"""
from __future__ import annotations

import numpy as np


def recovery_metrics(e_c, e_n, e_d) -> dict:
    """cos/MSE of noised & denoised pooled embeddings vs clean, and the fraction of the gap closed."""
    def _cos(a, b):
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
        return float((a * b).sum(1).mean())

    cos_n, cos_d = _cos(e_n, e_c), _cos(e_d, e_c)
    mse_n = float(((e_n - e_c) ** 2).mean())
    mse_d = float(((e_d - e_c) ** 2).mean())
    return {
        "cos_noised": cos_n, "cos_denoised": cos_d,
        "mse_noised": mse_n, "mse_denoised": mse_d,
        "recovery_cos": (cos_d - cos_n) / (1 - cos_n) if (1 - cos_n) > 1e-9 else 0.0,
        "recovery_mse": 1 - mse_d / mse_n if mse_n > 1e-12 else 0.0,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_snd.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/evals/snd_utility_sweep.py tests/test_snd.py
git commit -m "feat(eval): SnD utility-recovery metric helpers"
```

---

### Task 4: SnD sweep — capture, denoiser train, η sweep, JSON

**Files:**
- Modify: `scripts/evals/snd_utility_sweep.py` (add capture, training, main)

**Interfaces:**
- Consumes: `DxPrivacy`, `Denoiser` (Task 1/2), `recovery_metrics` (Task 3).
- Produces: a runnable CLI writing `refine-logs/snd/snd_utility_sweep.json`.

- [ ] **Step 1: Add capture + pooled-embedding + denoiser-training + main**

Append to `scripts/evals/snd_utility_sweep.py`:

```python
import argparse, json, math, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # scripts/ for defenses.*
from defenses.snd import DxPrivacy, Denoiser                          # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def capture_pooled(model, tok, prompts, eta, C, batch_size=32):
    """Mean-pooled last-hidden (e), padded input-embeddings X̃ (B,Tmax,d), lengths, and per-prompt
    token-id list. eta=inf ⇒ clean (no hook). Right-padded; pads masked from the pool + denoiser."""
    embs, Xt, lens = [], [], []
    hk = None if math.isinf(eta) else model.get_input_embeddings().register_forward_hook(DxPrivacy(C, eta))
    grab = {}
    h2 = model.get_input_embeddings().register_forward_hook(lambda m, i, o: grab.__setitem__("x", o.detach()))
    try:
        for i in range(0, len(prompts), batch_size):
            enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True)
            ids, mask = enc.input_ids.to(DEV), enc.attention_mask.to(DEV)
            out = model(ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
            last = out.hidden_states[-1].float()                       # (B,T,d) — post-hook embeds propagated
            m = mask.unsqueeze(-1).float()
            pooled = (last * m).sum(1) / m.sum(1).clamp_min(1.0)        # mean over real tokens
            xpad = grab["x"].float()                                   # (B,T,d): X̃ if hook else clean X
            for b in range(ids.shape[0]):
                n = int(mask[b].sum())
                embs.append(pooled[b].cpu().numpy())
                Xt.append(xpad[b, :n].cpu().numpy())
                lens.append(n)
    finally:
        h2.remove()
        if hk is not None:
            hk.remove()
    return np.stack(embs), Xt, lens


def _pad_batch(Xt_list, idx, d):
    """Stack a list of (n_i, d) arrays at positions `idx` into (B, Tmax, d) + bool pad mask (B,Tmax)."""
    sub = [Xt_list[i] for i in idx]
    Tmax = max(x.shape[0] for x in sub)
    B = len(sub)
    X = np.zeros((B, Tmax, d), np.float32)
    pad = np.ones((B, Tmax), bool)
    for b, x in enumerate(sub):
        X[b, :x.shape[0]] = x
        pad[b, :x.shape[0]] = False
    return X, pad


def train_denoiser(model, tok, prompts, train_etas, C, d, epochs, batch_size, seed):
    """Train ONE noise-aware denoiser on (e_n, X̃, Z, e_c) over several η. e_c captured once."""
    torch.manual_seed(seed)
    e_c, Xc, _ = capture_pooled(model, tok, prompts, math.inf, C, batch_size)   # clean pooled + clean X
    den = Denoiser(d=d).to(DEV)
    opt = torch.optim.Adam(den.parameters(), lr=1e-3)
    N = len(prompts)
    for ep in range(epochs):
        for eta in train_etas:
            e_n, Xt, _ = capture_pooled(model, tok, prompts, eta, C, batch_size)
            order = np.random.default_rng(seed + ep).permutation(N)
            for s in range(0, N, batch_size):
                idx = order[s:s + batch_size]
                Xtl, pad = _pad_batch(Xt, idx, d)
                Xcl, _ = _pad_batch(Xc, idx, d)
                Z = Xtl - Xcl
                ed = den(torch.from_numpy(e_n[idx]).to(DEV),
                         torch.from_numpy(Xtl).to(DEV), torch.from_numpy(Z).to(DEV),
                         pad_mask=torch.from_numpy(pad).to(DEV))
                loss = ((ed - torch.from_numpy(e_c[idx]).to(DEV)) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        print(f"[snd] denoiser epoch {ep+1}/{epochs} last-loss {loss.item():.4f}", flush=True)
    return den.eval(), e_c, Xc


@torch.no_grad()
def utility_pass(model, tok, prompts, eta, C, batch_size):
    """Teacher-forced perplexity + next-token top-1 acc under the dχ hook (eta=inf ⇒ clean)."""
    hk = None if math.isinf(eta) else model.get_input_embeddings().register_forward_hook(DxPrivacy(C, eta))
    ce_sum, n_tok, n_corr = 0.0, 0, 0
    try:
        for i in range(0, len(prompts), batch_size):
            enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True)
            ids, mask = enc.input_ids.to(DEV), enc.attention_mask.to(DEV)
            logits = model(ids, attention_mask=mask, use_cache=False).logits.float()
            pred, tgt, m = logits[:, :-1], ids[:, 1:], mask[:, 1:].bool()
            lp = torch.log_softmax(pred, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            ce_sum += float(-(lp[m]).sum()); n_tok += int(m.sum())
            n_corr += int((pred.argmax(-1)[m] == tgt[m]).sum())
    finally:
        if hk is not None:
            hk.remove()
    return math.exp(ce_sum / max(1, n_tok)), n_corr / max(1, n_tok)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--corpus", default="corpora/rep2text-stratified.txt")
    ap.add_argument("--max-prompts", type=int, default=400)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--etas", default="inf,100,50,10,1", help="dχ budget sweep; 'inf'=clean")
    ap.add_argument("--train-etas", default="50,10,1", help="η values the denoiser trains on")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--clip-percentile", type=float, default=99.9)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260626)
    ap.add_argument("--out", default="refine-logs/snd/snd_utility_sweep.json")
    args = ap.parse_args()

    etas = [math.inf if s.strip().lower().startswith("inf") else float(s)
            for s in args.etas.split(",") if s.strip()]
    if not any(math.isinf(e) for e in etas):
        etas = [math.inf] + etas
    train_etas = [float(s) for s in args.train_etas.split(",") if s.strip()]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    prompts = [l.strip() for l in Path(args.corpus).read_text().splitlines() if l.strip()][: args.max_prompts]
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, attn_implementation="eager", device_map=DEV).eval()
    d = model.config.hidden_size

    cal = []
    h = model.get_input_embeddings().register_forward_hook(
        lambda m, i, o: cal.append(o.float().norm(dim=-1).flatten().cpu()))
    try:
        with torch.no_grad():
            for p in prompts[:48]:
                model(tok(p, return_tensors="pt").input_ids.to(DEV), use_cache=False)
    finally:
        h.remove()
    C = float(np.percentile(torch.cat(cal).numpy(), args.clip_percentile))

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(prompts))
    ntr = int(args.train_frac * len(prompts))
    tr_prompts = [prompts[i] for i in perm[:ntr]]
    te_prompts = [prompts[i] for i in perm[ntr:]]
    print(f"[snd] model={args.model} d={d} C={C:.3f} etas={etas} train_etas={train_etas} "
          f"train/test={len(tr_prompts)}/{len(te_prompts)} dev={DEV}", flush=True)

    den, _, _ = train_denoiser(model, tok, tr_prompts, train_etas, C, d, args.epochs, args.batch_size, args.seed)

    e_c, Xc, _ = capture_pooled(model, tok, te_prompts, math.inf, C, args.batch_size)   # clean test
    base_ppl, base_acc = utility_pass(model, tok, te_prompts, math.inf, C, args.batch_size)
    records = []
    for eta in etas:
        e_n, Xt, _ = capture_pooled(model, tok, te_prompts, eta, C, args.batch_size)
        # denoise in batches (reuse padding helper)
        e_d = np.zeros_like(e_n)
        with torch.no_grad():
            for s in range(0, len(te_prompts), args.batch_size):
                idx = np.arange(s, min(s + args.batch_size, len(te_prompts)))
                Xtl, pad = _pad_batch(Xt, idx, d)
                Xcl, _ = _pad_batch(Xc, idx, d)
                Z = Xtl - Xcl
                ed = den(torch.from_numpy(e_n[idx]).to(DEV),
                         torch.from_numpy(Xtl).to(DEV), torch.from_numpy(Z).to(DEV),
                         pad_mask=torch.from_numpy(pad).to(DEV))
                e_d[idx] = ed.cpu().numpy()
        rec = {"eta": (None if math.isinf(eta) else eta)}
        rec.update(recovery_metrics(e_c, e_n, e_d))
        ppl, acc = utility_pass(model, tok, te_prompts, eta, C, args.batch_size)
        rec.update({"perplexity": ppl, "acc": acc,
                    "ppl_degradation": ppl / base_ppl - 1, "retention_acc": acc / base_acc if base_acc else None})
        records.append(rec)
        es = "inf" if math.isinf(eta) else f"{eta:g}"
        print(f"[snd] η={es:>5} | cos {rec['cos_noised']:.3f}->{rec['cos_denoised']:.3f} "
              f"rec_cos={rec['recovery_cos']:.3f} rec_mse={rec['recovery_mse']:.3f} "
              f"ppl_deg={rec['ppl_degradation']:+.2%}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "model": args.model, "corpus": args.corpus, "n_test": len(te_prompts), "hidden": d,
        "defense": "snd_dx", "clip_C": C, "clip_percentile": args.clip_percentile,
        "budget_note": "eta is the dχ-privacy budget (larger=weaker privacy); NOT the Gaussian ε "
                       "of dp_leakage_sweep — the two are not interchangeable.",
        "etas": [None if math.isinf(e) else e for e in etas], "train_etas": train_etas,
        "epochs": args.epochs, "seed": args.seed,
        "readout_note": "recovery_cos = fraction of (1-cos) gap closed by the denoiser; recovery_mse "
                        "= fraction of noised MSE removed; ppl/acc degradation = generation-utility "
                        "cost of the dχ noise (denoiser does not touch logits).",
        "records": records,
    }, indent=2))
    print(f"[snd] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it imports and arg-parses (CPU, no run)**

Run: `.venv/bin/python -c "import sys; sys.argv=['x','--help']; exec(open('scripts/evals/snd_utility_sweep.py').read())" 2>&1 | head -5`
Expected: argparse help text prints (no import/syntax error).

- [ ] **Step 3: Re-run the unit tests (helpers unchanged)**

Run: `.venv/bin/python -m pytest tests/test_snd.py -v`
Expected: PASS (7 tests)

- [ ] **Step 4: Commit**

```bash
git add scripts/evals/snd_utility_sweep.py
git commit -m "feat(eval): SnD utility-recovery sweep (dχ capture + denoiser train + η sweep)"
```

---

### Task 5: GPU smoke + real run (ROCm container)

**Files:**
- Output: `refine-logs/snd/snd_utility_sweep.json`

- [ ] **Step 1: GPU availability sanity**

Run: `scripts/run_in_rocm.sh python3 -c 'import torch; print(torch.cuda.is_available())'`
Expected: `True`

- [ ] **Step 2: Tiny smoke run (verify end-to-end, small scope)**

Run:
```bash
scripts/run_in_rocm.sh python3 scripts/evals/snd_utility_sweep.py \
    --max-prompts 40 --train-frac 0.5 --etas inf,50,1 --train-etas 10 --epochs 1 \
    --out refine-logs/snd/_smoke.json
```
Expected: prints per-η lines; writes `_smoke.json`; `cos_denoised` finite; no crash. Confirm wall-time, then delete `_smoke.json`.

- [ ] **Step 3: Real run (kill stray containers first; one GPU process)**

Run:
```bash
scripts/run_in_rocm.sh python3 scripts/evals/snd_utility_sweep.py \
    --etas inf,100,50,10,1,0.1 --train-etas 50,10,1 --epochs 2 \
    --out refine-logs/snd/snd_utility_sweep.json
```
Expected: monotone-ish `ppl_degradation` rising as η falls; `recovery_cos > 0` at noised η (denoiser helps). If recovery ≤ 0 at all η → that is the finding (note it; queue per-η-group denoiser per the spec's skipped list).

- [ ] **Step 4: Commit results**

```bash
git add refine-logs/snd/snd_utility_sweep.json
git commit -m "results(snd): utility-recovery vs dχ budget η (pythia-160m)"
```

---

## Self-Review

**Spec coverage:** DxPrivacy (Task 1) ✓ · Denoiser (Task 2) ✓ · pooled e_c/e_n/e_d + cos/MSE recovery (Tasks 3–4) ✓ · perplexity/acc degradation (Task 4) ✓ · η sweep + JSON + budget_note (Task 4) ✓ · GPU/perf gate (Task 5) ✓ · runnable self-checks (Tasks 1–3 pytest, replacing the spec's `__main__` demo with pytest — cleaner, same coverage) ✓. Privacy axis intentionally out of scope (spec) ✓.

**Placeholder scan:** none — every code step is complete.

**Type consistency:** `DxPrivacy(C,eta)` hook signature, `Denoiser(d,n_layers,n_heads).forward(e_n,X_tilde,Z,pad_mask)`, `recovery_metrics(e_c,e_n,e_d)` keys, and `_pad_batch`/`capture_pooled` returns are used consistently across Tasks 1–5.

**Note vs spec:** the spec named a `snd.py __main__` self-check; this plan uses `tests/test_snd.py` (pytest) for the same four checks — the host `.venv` runs them model-free. Equivalent coverage, repo-standard test location.
