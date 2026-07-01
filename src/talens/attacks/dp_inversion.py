"""Array-interface inversion attacks for the DP / cross-layer leakage sweeps.

These are the attacks `scripts/evals/dp_leakage_sweep.py` (and future BeamClean) call: each maps a
captured representation matrix to predicted token ids via the embedding table, on a plain numpy
(Xtr, Etr, Xte, pool_emb, pool_ids) interface — deliberately lighter than the CaptureSet/AttackResult
API in `hidden_state.py` / `_inversion.py`, which target the full library pipeline.

Repo rule: attacks live here (src/talens/attacks/), not in scripts. evals/spikes only call them.

  ridge_attack         — linear (ridge) obs→embedding map, then nearest token (the strong
                         single-position baseline; ≈ tuned-lens affine-into-E).
  skip_decoder_attack  — gated linear-skip GELU decoder, ridge-warm-started + frozen linear path,
                         early-stopped on a disjoint val split → clean `decoder ≥ ridge`.
  logit_lens_attack    — CE logit-lens: query head (affine [+ gated GELU]) decoded through the frozen
                         embedding table with cross-entropy over full/sampled vocab; the objective-
                         matched, fair non-linearity test. nonlinear=False ⇒ tuned-lens affine.
                         See docs/research/ce-logit-lens-attack.md.
  (BeamClean LM-prior beam decode will land here as beamclean_attack — campaign-D Task 5.)
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def nearest_token(pred_emb, pool_emb, pool_ids):
    """Cosine nearest-neighbour decode of predicted embeddings against a candidate pool."""
    p = pred_emb / np.clip(np.linalg.norm(pred_emb, axis=1, keepdims=True), 1e-9, None)
    e = pool_emb / np.clip(np.linalg.norm(pool_emb, axis=1, keepdims=True), 1e-9, None)
    return pool_ids[(p @ e.T).argmax(1)]


def ridge_W(Xtr, Etr, alpha=1.0):
    """Closed-form ridge map X→E (float64 Gram/solve for stability; returned float32 [d_in, d_out])."""
    d = Xtr.shape[1]
    A = (Xtr.T @ Xtr).astype(np.float64) + alpha * np.eye(d, dtype=np.float64)
    return np.linalg.solve(A, (Xtr.T @ Etr).astype(np.float64)).astype(np.float32)


def multikey_ridge_W(G0, H0, Pks, alpha=1.0):
    """Multi-key-synthesis ridge (AloePri ISA-HiddenState blind, paper §F.1). Fits one inverter over
    pooled synthetic obfuscated reps {Xc·Pk}_k WITHOUT materializing the K·n stack:

        X^T X = Σ_k Pk^T G0 Pk,   X^T E = (Σ_k Pk)^T H0,   with G0 = Xc^T Xc, H0 = Xc^T E.

    Algebraically identical to stacking and ridge_W (test_multikey_ridge_matches_stacking); K small
    matmuls instead of a K·n-row solve. Returns the (d+2h)→d_emb map (float32). float64 internally."""
    D1 = Pks[0].shape[1]
    Gram = np.zeros((D1, D1), np.float64)
    Psum = np.zeros_like(Pks[0], dtype=np.float64)
    for Pk in Pks:
        Pk64 = np.asarray(Pk, np.float64)
        Gram += Pk64.T @ np.asarray(G0, np.float64) @ Pk64
        Psum += Pk64
    Gram += alpha * np.eye(D1)
    return np.linalg.solve(Gram, Psum.T @ np.asarray(H0, np.float64)).astype(np.float32)


def ridge_attack(Xtr, Etr, Xte, pool_emb, pool_ids, *, alpha=1.0, **_):
    """Linear (ridge) obs→embedding map, then nearest token. The strong single-position baseline."""
    return nearest_token(Xte @ ridge_W(Xtr, Etr, alpha), pool_emb, pool_ids)


def orthogonal_procrustes_R(P, D):
    """Orthogonal R ∈ O(d) minimising ‖P·R − D‖_F — closed form R = U Vᵀ from svd(Pᵀ D).
    So P·R ≈ D; map D back into the P basis with D·Rᵀ. float64 internally, returns float32 (d,d).
    This is the least-squares known-plaintext solution for an orthogonal cipher (needs ~d anchors)."""
    M = np.asarray(P, np.float64).T @ np.asarray(D, np.float64)
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return (U @ Vt).astype(np.float32)


def blockwise_procrustes_R(P, D, n_heads=12, hd=64):
    """Per-head block variant of orthogonal_procrustes_R for R = head-permutation ∘ blkdiag(Û_vo)
    (AloePri Alg2's value transform). Fit an O(hd) Procrustes per (plaintext head h, deployment head h'),
    assign the head-permutation by min total residual (Hungarian if scipy, else greedy), assemble the
    block-permuted-orthogonal R (P·R ≈ D). Needs only ~hd anchors (each anchor supplies all heads), vs
    ~n_heads·hd for the global solve — the sample-efficient known-plaintext solve when R is block-structured."""
    Pb = np.asarray(P, np.float64).reshape(-1, n_heads, hd)
    Db = np.asarray(D, np.float64).reshape(-1, n_heads, hd)
    res = np.zeros((n_heads, n_heads)); Q = {}
    for h in range(n_heads):
        for hp in range(n_heads):
            U, _, Vt = np.linalg.svd(Pb[:, h].T @ Db[:, hp], full_matrices=False)
            Q[(h, hp)] = U @ Vt
            res[h, hp] = np.linalg.norm(Pb[:, h] @ Q[(h, hp)] - Db[:, hp])
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(res)
    except Exception:                                        # greedy fallback (few heads)
        rows, cols, used = list(range(n_heads)), [], set()
        for h in range(n_heads):
            hp = next(c for c in np.argsort(res[h]) if c not in used); used.add(hp); cols.append(hp)
    R = np.zeros((n_heads * hd, n_heads * hd), np.float32)
    for h, hp in zip(rows, cols):
        R[h * hd:(h + 1) * hd, hp * hd:(hp + 1) * hd] = Q[(h, hp)].astype(np.float32)
    return R


def rotation_recovery_attack(Xp_align, Xd_align, Xp_tr, ytr, Xd_te, pool_emb, pool_ids, *,
                             table, alpha=1.0, R_estimator=orthogonal_procrustes_R, **_):
    """Known-plaintext rotation-recovery inversion (claim:aloepri-kqvout-basis-alignment). A KNOWN attack
    (orthogonal-Procrustes known-plaintext recovery of a linear/orthogonal obfuscation — cf. cross-lingual
    embedding alignment MUSE/VecMap, Hill-cipher KPA, and the orthogonal-obfuscation LLM-inference line
    arXiv:2606.16461 / 2603.01499), instantiated on AloePri Alg2's per-head value rotation of kqv_out.

    AloePri Alg2 rotates kqv_out by a SECRET but context-independent (block-)orthogonal map R
    (deployment = plaintext·R). A keyless self-generated inverter (§05) collapses under Alg2 because it
    cannot reproduce R. This attack spends a harvest on recovering R ONLY — not on the token map:

      1. estimate R̂ from aligned (plaintext, deployment) anchor pairs (`Xp_align`, `Xd_align`) via
         `R_estimator` (orthogonal_procrustes_R global, or blockwise_procrustes_R per-head);
      2. un-rotate the deployment reps into the plaintext basis (`Xd_te · R̂ᵀ`);
      3. decode with a self-generated ridge fit on the attacker's own plaintext reps (`Xp_tr → emb[ytr]`).

    Keyless: R̂ from the (threat-model-legitimate) anchors, token map from unlimited self-generation."""
    R = R_estimator(Xp_align, Xd_align)                      # Xp·R ≈ Xd
    W = ridge_W(Xp_tr, table[ytr], alpha)                    # self-gen inverter, plaintext basis
    return nearest_token((Xd_te @ R.T) @ W, pool_emb, pool_ids)


def cascade_attack(attack, X, y, harvested_types, table, pool, *, X_aug=None, y_aug=None, **kw):
    """Two-stage τ-leak cascade. A harvest (e.g. TFMA) reveals the true labels for `harvested_types`
    (a set/array of token ids); train ANY array-interface `attack` on those (deployment-basis rep,
    token) pairs — optionally augmented with blind pairs (`X_aug`, `y_aug`, e.g. multi-key synthetic
    reps) — and score recovery on the HELD-OUT (unharvested) types. Generic over the target
    representation: `X` / `table` / `attack` decide whether it is embeddings, residual, or q/k/v
    (residual ISA-HiddenState, IMA-EmbedRow on the static table, … all reuse this).

    Returns {unharvested (generalization to never-harvested types), harvested (in-set sanity),
    n_harv_types, n_held}. `unharvested` is the bootstrap signal: does knowing k token mappings let the
    inverter read the rest? `**kw` is forwarded to `attack` (alpha / hidden / epochs / seed / …)."""
    H = {int(t) for t in harvested_types}
    inset = np.fromiter((int(t) in H for t in y), bool, len(y))
    tr, te = np.where(inset)[0], np.where(~inset)[0]
    if te.size == 0 or (tr.size == 0 and X_aug is None):
        return {"unharvested": None, "harvested": None, "n_harv_types": len(H), "n_held": int(te.size)}
    Xtr, ytr = X[tr], y[tr]
    if X_aug is not None:
        Xtr = np.concatenate([Xtr, X_aug], 0)
        ytr = np.concatenate([ytr, np.asarray(y_aug)])
    pred = lambda idx: attack(Xtr, table[ytr], X[idx], table[pool], pool, ytr=ytr, full_emb=table, **kw)
    return {"unharvested": float((pred(te) == y[te]).mean()),
            "harvested": (float((pred(tr) == y[tr]).mean()) if tr.size else None),
            "n_harv_types": int(np.unique(y[tr]).size if tr.size else 0), "n_held": int(te.size)}


class _IMABlock(torch.nn.Module):
    """One inverter block: pre-LN MultiheadAttention (seq=1 ⇒ ≈ identity) + pre-LN GELU FFN, residual."""

    def __init__(self, hidden, n_heads, ffn_mult=4):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(hidden)
        self.attn = torch.nn.MultiheadAttention(hidden, n_heads, batch_first=True, bias=False)
        self.ln2 = torch.nn.LayerNorm(hidden)
        self.ffn = torch.nn.Sequential(torch.nn.Linear(hidden, hidden * ffn_mult, bias=False),
                                       torch.nn.GELU(), torch.nn.Linear(hidden * ffn_mult, hidden, bias=False))

    def forward(self, x):
        xn = self.ln1(x)
        x = x + self.attn(xn, xn, xn, need_weights=False)[0]
        return x + self.ffn(self.ln2(x))


class IMAInverter(torch.nn.Module):
    """AloePri IMA paper §F.1 inverter: input proj → n_layers×(MHA+GELU-FFN) → output proj.
    Port of private-rag run_ima_paper_like.IMAInverter (2 layers, 8 heads). For the static embedding
    table each row is its own sequence (seq=1), so the attention is near-identity and the block is an
    LN+GELU-FFN MLP; the architecture is kept faithful so the L0/sequence variant can reuse it."""

    def __init__(self, observed_dim, output_dim, hidden=768, n_layers=2, n_heads=8, ffn_mult=4):
        super().__init__()
        self.input_proj = torch.nn.Linear(observed_dim, hidden, bias=False) if observed_dim != hidden else torch.nn.Identity()
        self.blocks = torch.nn.ModuleList([_IMABlock(hidden, n_heads, ffn_mult) for _ in range(n_layers)])
        self.output_proj = torch.nn.Linear(hidden, output_dim, bias=False)

    def forward(self, x):
        h = self.input_proj(x)
        for blk in self.blocks:
            h = blk(h)
        return self.output_proj(h)


def ima_transformer_attack(Xtr, Etr, Xte, pool_emb, pool_ids, *, hidden=768, n_layers=2, n_heads=8,
                           epochs=12, lr=1e-3, batch=4096, seed=0, **_):
    """AloePri IMA-EmbedRow-transformer (paper §F.1): train the 2-layer/8-head IMAInverter on
    (observed-row → plain-embedding) pairs with MSE, then decode test rows by nearest-token. The caller
    decides the threat model via what it passes as Xtr/Etr: a SINGLE key (matched / plain control) or
    many synthetic own-key obfuscations (τ-invariant, the in-model attack). Rows are independent (seq=1)."""
    torch.manual_seed(seed)
    net = IMAInverter(Xtr.shape[1], Etr.shape[1], hidden=hidden, n_layers=n_layers, n_heads=n_heads).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    Xt = torch.from_numpy(np.ascontiguousarray(Xtr)).to(DEV)
    Et = torch.from_numpy(np.ascontiguousarray(Etr)).to(DEV)
    n = Xt.shape[0]
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            pred = net(Xt[idx].unsqueeze(1)).squeeze(1)          # (b,1,d)->(b,d)
            torch.nn.functional.mse_loss(pred, Et[idx]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        out = net(torch.from_numpy(np.ascontiguousarray(Xte)).to(DEV).unsqueeze(1)).squeeze(1).cpu().numpy()
    return nearest_token(out, pool_emb, pool_ids)


def nn_attack(Xtr, Etr, Xte, pool_emb, pool_ids, **_):
    """NN — Nearest-Neighbour attack (AloePri paper §F.1 / Table 1). TRAINING-FREE: cosine-match each
    observed hidden-state row directly to the nearest candidate token-embedding row. Recovers at L0 /
    plaintext (residual ≈ embedding) but needs the observation to live in the embedding space — under
    AloePri the released residual is in the secret P̂-basis (and a different width), so a cross-space
    match is undefined and recovery is chance (paper reports AloePri NN = 0%). Xtr/Etr are unused."""
    if Xte.shape[1] != pool_emb.shape[1]:        # obf basis (d+2h) ≠ embedding space → cannot match
        return np.full(Xte.shape[0], pool_ids[0], dtype=pool_ids.dtype)   # degenerate ⇒ ~chance
    return nearest_token(Xte, pool_emb, pool_ids)


class LinearSkipDecoder(torch.nn.Module):
    """Gated linear-skip decoder: pred = Linear(x) + gate · MLP(x), GELU, narrow hidden.

    The linear path is warm-started to ridge and FROZEN (carries the strong affine/tuned-lens
    baseline); the GELU MLP adds a gated non-linear correction with gate init 0 (ReZero-style — the
    MLP tail keeps its normal init so the gate has a live gradient; zeroing BOTH gate and tail
    pins the branch at a zero-gradient saddle ≡ ridge), so the decoder starts identical to ridge
    and the non-linearity switches on only if the held-out split supports it. GELU (not ReLU)
    avoids dead-neuron narrowing; a NARROW hidden (≤ input) + early-stop control overfitting in the
    small-data regime. See research-wiki/claims/single-position-residual-linearly-saturated.md.
    """

    def __init__(self, d_in, d_out, hidden):
        super().__init__()
        self.lin = torch.nn.Linear(d_in, d_out)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, d_out)
        )
        self.gate = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.lin(x) + self.gate * self.mlp(x)


def skip_decoder_attack(Xtr, Etr, Xte, pool_emb, pool_ids, *, hidden=384, epochs=500, lr=1e-3,
                        seed=0, val_frac=0.15, patience=40, **_):
    """Gated linear-skip GELU decoder, ridge-warm-started + FROZEN linear path, early-stopped.

    The linear path is set to ridge's exact W and frozen; only the gated GELU correction trains
    (gate=0 at init → starts identical to ridge; ReZero-style, so the MLP tail keeps its normal
    init and the gate has a live gradient), with weight-decay + early-stopping on a held-out
    split. So the decoder starts identical to ridge and can only add a data-supported non-linear
    correction — a clean `decoder ≥ ridge` guarantee with overfitting control.
    """
    torch.manual_seed(seed)
    net = LinearSkipDecoder(Xtr.shape[1], Etr.shape[1], hidden).to(DEV)
    with torch.no_grad():  # warm-start: linear path = ridge, gate=0 → forward starts AT ridge
        net.lin.weight.copy_(torch.from_numpy(np.ascontiguousarray(ridge_W(Xtr, Etr).T)).to(DEV))
        net.lin.bias.zero_()
        # ReZero-style: gate=0 alone makes the net start at ridge (gate·mlp=0). The MLP tail is
        # left at its NORMAL init — do NOT zero it, or ∂loss/∂gate = mlp(x) = 0 and the whole
        # non-linear branch is a zero-gradient saddle that Adam can never leave (pins ≡ ridge).
    for p in net.lin.parameters():  # freeze the ridge baseline; train only the gated correction
        p.requires_grad_(False)

    def _unit(A):
        t = torch.from_numpy(A).to(DEV)
        return t / t.norm(dim=1, keepdim=True).clamp_min(1e-9)

    perm = np.random.default_rng(seed).permutation(Xtr.shape[0])
    nval = max(1, int(val_frac * Xtr.shape[0]))
    vi, ti = perm[:nval], perm[nval:]
    xt, yt = torch.from_numpy(Xtr[ti]).to(DEV), _unit(Etr[ti])
    xv, yv = torch.from_numpy(Xtr[vi]).to(DEV), _unit(Etr[vi])
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=1e-3)
    best, best_state, bad = float("inf"), copy.deepcopy(net.state_dict()), 0  # init state = ridge
    for _ in range(epochs):
        net.train()
        opt.zero_grad()
        p = net(xt)
        p = p / p.norm(dim=1, keepdim=True).clamp_min(1e-9)
        (1.0 - (p * yt).sum(1)).mean().backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            pv = net(xv)
            pv = pv / pv.norm(dim=1, keepdim=True).clamp_min(1e-9)
            vloss = (1.0 - (pv * yv).sum(1)).mean().item()
        if vloss < best - 1e-5:
            best, best_state, bad = vloss, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)  # best held-out state (ridge-at-init is in the running → ≥ ridge)
    net.eval()
    with torch.no_grad():
        pred = net(torch.from_numpy(Xte).to(DEV)).cpu().numpy()
    return nearest_token(pred, pool_emb, pool_ids)


def isa_grad_attack(Xtr, Etr, Xte, pool_emb, pool_ids, *, alpha=1.0, steps=500, lr=5e-2, seed=0, **_):
    """AloePri §D.1 gradient-optimization ISA — analysis-by-synthesis on input embeddings.

    Ridge fits the DISCRIMINATIVE obs→embedding map and decodes in one closed form. §D.1 instead
    fits the GENERATIVE forward-synthesis map Ŵ_g: clean-embedding → observed deployment rep from
    the harvested pairs (= the obfuscation P̂ at L0), then RECOVERS each held-out observed rep x̃ by
    GRADIENT-OPTIMIZING an input embedding e to minimise ‖e·Ŵ_g − x̃‖² (Adam from e=0), nearest-token e.

    "ridge bypasses the optimization landscape" (private-rag 2B.1 doc, §D.1): this runs the actual
    descent over input-embedding space rather than a closed-form inverse. With a LINEAR synthesis map
    the landscape is convex, so the descent lands at the generative least-squares inverse
    e* = x̃·Ŵ_g⁺ — which still differs from ridge's discriminative map when the obfuscation WIDENS the
    basis (d̃ = d+2h > d_emb, so e·Ŵ_g = x̃ is over-determined) or under αₑ output noise (synthesis
    models it as output noise, the physically-correct direction; ridge folds it into the input). Whether
    that gap survives a harvest — i.e. whether gradient-opt ISA beats or merely equals ridge — is the test.
    """
    torch.manual_seed(seed)
    Wg = torch.from_numpy(ridge_W(Etr, Xtr, alpha)).to(DEV)            # clean emb → obs rep (d_emb, d̃)
    Xt = torch.from_numpy(np.ascontiguousarray(Xte)).to(DEV)
    e = torch.zeros(Xt.shape[0], Wg.shape[0], device=DEV, requires_grad=True)
    opt = torch.optim.Adam([e], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        F.mse_loss(e @ Wg, Xt).backward()
        opt.step()
    return nearest_token(e.detach().cpu().numpy(), pool_emb, pool_ids)


class LensHead(torch.nn.Module):
    """Residual→query head for the CE logit lens: q = A·x + b [+ gate·GELU-MLP(x)].

    Logits are formed downstream as E·q against the FROZEN embedding table (no per-token params), so
    the head only learns a query map. ReZero init for the non-linear branch (gate=0 but MLP tail at
    normal init → live gate gradient). See docs/research/ce-logit-lens-attack.md.
    """

    def __init__(self, d_in, d_emb, hidden, nonlinear):
        super().__init__()
        self.lin = torch.nn.Linear(d_in, d_emb)
        self.nonlinear = nonlinear
        if nonlinear:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(d_in, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, d_emb)
            )
            self.gate = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        q = self.lin(x)
        if self.nonlinear:
            q = q + self.gate * self.mlp(x)
        return q


def logit_lens_attack(Xtr, Etr, Xte, pool_emb, pool_ids, *, ytr, full_emb, nonlinear=True,
                      neg=2048, hidden=384, epochs=500, lr=1e-3, seed=0, val_frac=0.15,
                      patience=40, **_):
    """CE logit-lens attack — objective-matched, fair non-linearity test (docs/research/ce-logit-lens-attack.md).

    Trains a query head h(x) scored against the frozen embedding table `full_emb` with COSINE-softmax
    cross-entropy (query + embeddings row-normalised, fixed temperature) over a SAMPLED full vocab
    (candidate set = unique train tokens ∪ `neg` random negatives, resampled per epoch). Early-stopped
    on val TOP-1 recovery (seeded with the ridge warm-start → ≥ ridge), decoded by COSINE over the test
    `pool` (identical convention to ridge). Cosine — not raw dot product — because gemma token-embedding
    norms are heterogeneous: dot-product logits rank by ‖E‖ and badly under-recover (init dot 0.78 vs
    cosine 1.0 on norm-heterogeneous synthetic); and val-top1 (not val-CE) early-stop keeps the metric
    aligned. `ytr` = train token ids (CE positives — pass the SAME shuffle as `Etr` for the floor);
    `Etr` warm-starts the affine to ridge. nonlinear=False ⇒ tuned-lens affine; True ⇒ + gated GELU.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d_in, d_emb, V = Xtr.shape[1], full_emb.shape[1], full_emb.shape[0]
    ytr = np.asarray(ytr).astype(np.int64)
    assert ytr.shape[0] == Xtr.shape[0] and (0 <= ytr).all() and (ytr < V).all(), \
        "ytr must be valid token ids aligned to Xtr (CE positives index full_emb)"
    net = LensHead(d_in, d_emb, hidden, nonlinear).to(DEV)
    with torch.no_grad():  # warm-start the affine path to ridge (≈ tuned-lens start)
        net.lin.weight.copy_(torch.from_numpy(np.ascontiguousarray(ridge_W(Xtr, Etr).T)).to(DEV))
        net.lin.bias.zero_()

    perm = rng.permutation(Xtr.shape[0])
    nval = max(1, int(val_frac * Xtr.shape[0]))
    vi, ti = perm[:nval], perm[nval:]
    xt, yt = torch.from_numpy(Xtr[ti]).to(DEV), ytr[ti]
    xv, yv = torch.from_numpy(Xtr[vi]).to(DEV), ytr[vi]
    Efull = torch.from_numpy(np.ascontiguousarray(full_emb))  # CPU; gather candidate rows per step
    train_toks = np.unique(ytr)  # always in the candidate set (CE positives must be in the denominator)

    def _unit(t):  # row-normalise → cosine geometry (decouples from gemma's heterogeneous token norms)
        return t / t.norm(dim=1, keepdim=True).clamp_min(1e-9)

    # COSINE softmax CE (normalise query + embeddings, fixed temperature) — matches the cosine decode;
    # raw dot-product logits would rank by ‖E‖ and badly under-recover (init dot 0.78 vs cosine 1.0).
    tau = 0.05
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-3)

    def _val_top1():  # early-stop METRIC = val top-1 recovery (train-token retrieval) — aligned with
        with torch.no_grad():  # the test metric, unlike val CE loss (which drifts off it)
            negs = rng.choice(V, size=min(neg, V), replace=False)
            cand = np.unique(np.concatenate([train_toks, negs]))
            Ec = _unit(Efull[cand].to(DEV))
            tgt = torch.from_numpy(np.searchsorted(cand, yv)).to(DEV)
            return float((( _unit(net(xv)) @ Ec.T).argmax(1) == tgt).float().mean())

    best, best_state, bad = _val_top1(), copy.deepcopy(net.state_dict()), 0  # seed with ridge init → ≥ ridge
    for _ in range(epochs):
        negs = rng.choice(V, size=min(neg, V), replace=False)
        # cand sorted+deduped; train_toks ⊆ cand so every positive yt/yv is in it →
        # searchsorted returns the EXACT class index (no off-by-one; collisions dedup'd).
        cand = np.unique(np.concatenate([train_toks, negs]))
        Ec = _unit(Efull[cand].to(DEV))
        tgt_t = torch.from_numpy(np.searchsorted(cand, yt)).to(DEV)
        net.train()
        opt.zero_grad()
        F.cross_entropy((_unit(net(xt)) @ Ec.T) / tau, tgt_t).backward()
        opt.step()
        net.eval()
        vrec = _val_top1()
        if vrec > best + 1e-5:
            best, best_state, bad = vrec, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():  # decode by COSINE over the test pool (identical convention to ridge)
        pred = net(torch.from_numpy(Xte).to(DEV)).cpu().numpy()
    return nearest_token(pred, pool_emb, pool_ids)
