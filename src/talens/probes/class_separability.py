"""Token-class separability on a (possibly DP-noised) representation surface.

Adapts the MDL-probe separability picture (Voita & Titov, arXiv:1909.01380 — "does the
representation sort the labels?") to the MI-vs-attack-recovery / DP setting: given a small fixed
set of closely-related token classes (e.g. the *to be* forms ``{is, are, was, were}``), measure how
distinguishable the class-conditional representation clouds are — **without training the
token-recovery attack**.

Two attack-independent numbers in bits (converse + achievable — the repo's standard pairing, cf.
``claim:bnn-error-bounds-bhattacharyya-fano``):

* **converse** — geometry-only Bhattacharyya/Fano on the class-conditional Gaussians under a
  homoscedastic LDA model: whiten by the pooled within-class covariance, then the class means + unit
  isotropic noise feed :mod:`talens.probes.channel_error_bounds`. No classifier is trained, so a
  depth peak is a real geometric fact, not the recovery attack in disguise.
* **achievable** — the canonical class-probe MDL online code length
  (:func:`talens.probes.mdl.online_code_length`) on the K-class task. Independent of the ridge
  VOCAB attack (a different, tiny label set), so it sidesteps the retrieval-MDL↔ridge circularity.

Resolves handoff open-#1 (I_G measures *representation*-survival, not *token*-survival at depth): a
separability trajectory that PEAKS at L20 ⇒ real token information the linear attack can't read; one
that FALLS monotonically ⇒ tracks recovery, exposing the I_G/CLUB L20 peak as a context artifact.
"""
from __future__ import annotations

import numpy as np

from .channel_error_bounds import fano_equivocation, union_bhattacharyya
from .mdl import online_code_length


def _pca_reduce(X: np.ndarray, dim: int) -> np.ndarray:
    """Center + project onto the top-``dim`` principal directions (cov-eigh).

    # ponytail: cov-eigh on the d×d covariance, not a full SVD — d is the model width and these
    # row counts are small, so CPU is fine; no GPU path warranted here.
    """
    Xc = X - X.mean(0)
    w, V = np.linalg.eigh(np.cov(Xc.T))
    comps = V[:, ::-1][:, :dim]            # top-dim eigenvectors (eigh is ascending)
    return Xc @ comps


def _whiten_within_class(Xr: np.ndarray, cls: np.ndarray, n_cls: int, ridge: float = 1e-3) -> np.ndarray:
    """Pooled within-class covariance whitening (homoscedastic LDA): within-class scatter → ~I."""
    d = Xr.shape[1]
    Sw = np.zeros((d, d))
    for c in range(n_cls):
        Z = Xr[cls == c]
        if Z.shape[0] < 2:
            continue
        Zc = Z - Z.mean(0)
        Sw += Zc.T @ Zc
    Sw /= max(1, Xr.shape[0] - n_cls)
    Sw += ridge * np.trace(Sw) / d * np.eye(d)        # PD floor
    w, V = np.linalg.eigh(0.5 * (Sw + Sw.T))
    w = np.clip(w, 1e-12, None)
    return Xr @ ((V / np.sqrt(w)) @ V.T)               # Xr · Sw^{-1/2}


def class_separability(
    X: np.ndarray,
    y: np.ndarray,
    class_words: dict[str, tuple[int, ...]],
    *,
    max_dim: int = 64,
    mdl_dim: int = 16,
    coords_cap: int = 120,
    seed: int = 0,
    want_coords: bool = True,
) -> dict:
    """Converse + achievable separability of the token classes in ``class_words``.

    ``X`` (N,d) rep rows; ``y`` (N,) token ids; ``class_words`` maps a class label to the token
    id(s) that belong to it. Rows whose token id is in no class are dropped.

    Readouts (the token classes are usually trivially Bayes-separable, so the channel-MI converse
    saturates at ``log₂K`` and carries no trajectory — the *margin* is what moves):

    * ``bits`` (headline) — MDL achievable info ``uniform − online code`` (bits, leakage-positive,
      has dynamic range). Probe runs in a low ``mdl_dim`` subspace so the prequential blocks don't
      overfit on the minority classes.
    * ``bhat_dist`` — mean pairwise Bhattacharyya distance of the (whitened) class Gaussians
      (unbounded separability margin in nats); the geometry-only trajectory.
    * ``mi_converse_bits`` — capped channel-MI ``log₂K − H(V|Y)`` (the "fully separable ceiling").
    * ``p_e_ub`` / ``p_e_lb`` — Bayes-error bounds; 2D PCA ``coords`` for the cloud figure.
    """
    id2cls: dict[int, int] = {}
    labels: list[str] = []
    for ci, (lab, ids) in enumerate(class_words.items()):
        labels.append(lab)
        for t in ids:
            id2cls[int(t)] = ci

    sel = np.array([int(t) in id2cls for t in y])
    if sel.sum() < 8:
        return {"bits": None, "bits_kind": "class_separability",
                "note": "too few class rows", "n_rows": int(sel.sum())}

    Xs = X[sel].astype(np.float64)
    cls0 = np.array([id2cls[int(t)] for t in y[sel]])
    present = np.unique(cls0)
    K = int(present.size)
    if K < 2:
        return {"bits": None, "bits_kind": "class_separability",
                "note": "<2 classes present", "k_present": K, "n_rows": int(sel.sum())}
    remap = {int(c): i for i, c in enumerate(present)}
    cls = np.array([remap[int(c)] for c in cls0])

    dim = max(2, min(max_dim, Xs.shape[1], Xs.shape[0] - 1))
    Xr = _pca_reduce(Xs, dim)

    # converse — whiten within-class, then class means + unit isotropic noise feed Bhattacharyya/Fano
    Xw = _whiten_within_class(Xr, cls, K)
    means = np.stack([Xw[cls == c].mean(0) for c in range(K)])
    ub = union_bhattacharyya(means, sigma=1.0)
    fa = fano_equivocation(means, sigma=1.0, M=128, seed=seed)   # i_channel_bits None when K<3
    i_conv = fa.get("i_channel_bits")
    # Bhattacharyya distance margin (homoscedastic, Σ=I after whitening ⇒ D_B = ⅛‖Δμ‖²): unbounded,
    # so unlike the capped channel-MI it keeps a trajectory even when the classes are fully separable.
    mu_sq = ((means[:, None, :] - means[None, :, :]) ** 2).sum(-1)
    iu = np.triu_indices(K, k=1)
    bhat_dist = float(np.mean(mu_sq[iu]) / 8.0) if iu[0].size else None

    # achievable — class-probe MDL in a LOW-dim subspace (prequential blocks overfit in 64-d on the
    # minority classes); online_code_length remaps the int labels to 0..K-1 itself.
    mdl = online_code_length(Xr[:, : min(mdl_dim, Xr.shape[1])], cls, seed=20260615 + seed)
    mdl_info = None
    if mdl.get("online_code_length_bits") is not None and mdl.get("uniform_code_length_bits") is not None:
        mdl_info = float(mdl["uniform_code_length_bits"] - mdl["online_code_length_bits"])

    out = {
        "bits": mdl_info,                                    # headline = MDL achievable info (has range)
        "bits_kind": "class_separability",
        "bhat_dist": bhat_dist,                              # geometry-only separability margin (nats)
        "mi_converse_bits": None if i_conv is None else float(i_conv),
        "p_e_ub": ub.get("p_e_ub"), "p_e_lb": fa.get("p_e_lb"),
        "mdl_info_bits": mdl_info, "mdl_code_bits": mdl.get("online_code_length_bits"),
        "compression": mdl.get("compression"),
        "k_present": K, "n_rows": int(sel.sum()),
        "labels": [labels[int(c)] for c in present],
        "per_class_n": {labels[int(present[i])]: int((cls == i).sum()) for i in range(K)},
    }
    if want_coords:
        rng = np.random.default_rng(seed)
        pts = []
        for c in range(K):
            idx = np.where(cls == c)[0]
            if idx.size > coords_cap:
                idx = rng.choice(idx, coords_cap, replace=False)
            lab = labels[int(present[c])]
            pts.extend((float(Xr[j, 0]), float(Xr[j, 1]), lab) for j in idx)
        out["coords"] = pts
    return out


def demo() -> None:
    """Self-check: well-separated classes give high converse+MDL bits; overlapping give ~0."""
    rng = np.random.default_rng(0)
    d, n = 32, 300
    # token ids 101,102,103 are the three classes; build separable vs overlapping reps.
    y = np.repeat([101, 102, 103], n)
    cw = {"a": (101,), "b": (102,), "c": (103,)}

    centers = rng.standard_normal((3, d)) * 6.0
    Xsep = np.concatenate([centers[c] + rng.standard_normal((n, d)) for c in range(3)])
    Xover = rng.standard_normal((3 * n, d))   # all classes from one cloud → not separable

    rsep = class_separability(Xsep, y, cw, seed=0)
    rover = class_separability(Xover, y, cw, seed=0)

    assert rsep["bits"] > rover["bits"] + 0.5, f"MDL info separable !> overlapping: {rsep['bits']} vs {rover['bits']}"
    assert rsep["bhat_dist"] > rover["bhat_dist"] * 5, f"margin separable !>> overlapping: {rsep['bhat_dist']} vs {rover['bhat_dist']}"
    assert rsep["mi_converse_bits"] > rover["mi_converse_bits"] + 0.5, "channel-MI should be higher when separable"
    assert rsep["p_e_ub"] < rover["p_e_ub"], "Bayes-error UB should be lower when separable"
    assert len(rsep["coords"]) > 0 and rsep["k_present"] == 3
    print(f"OK  separable: mdl={rsep['bits']:.0f}b margin={rsep['bhat_dist']:.1f} mi={rsep['mi_converse_bits']:.2f}b "
          f"p_e_ub={rsep['p_e_ub']:.3f} | overlapping: mdl={rover['bits']:.0f}b margin={rover['bhat_dist']:.2f} "
          f"mi={rover['mi_converse_bits']:.2f}b p_e_ub={rover['p_e_ub']:.3f}")


if __name__ == "__main__":
    demo()
