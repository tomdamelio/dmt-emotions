"""Cluster-based permutation tests for High-vs-Low time courses (revision, R2.2).

Reviewer 2 asked for a continuous clustering approach instead of the
window-by-window paired t-tests + BH-FDR used in the submitted manuscript.
This module runs three inferences on the SAME subject x window matrices the
published figures use, so results are directly comparable:

  fdr      -- the published method, reimplemented here as a control. Must
              reproduce the published shaded segments before anything else
              is interpreted.
  cluster  -- sign-flip permutation with a fixed cluster-forming threshold
              (t at uncorrected p < .05, df = n-1), cluster mass = sum of t.
  tfce     -- the same permutation with threshold-free cluster enhancement
              (E = 0.5, H = 2, steps of 0.1), which removes the arbitrary
              cluster-forming threshold and returns a p per window.

The permutation is exhaustive (2**n sign flips), so p-values are exact.
Time is 1-D, so adjacency is simply "neighbouring windows".

Nothing here modifies the published scripts or results.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

from mne.stats import permutation_cluster_1samp_test

ALPHA = 0.05
TFCE_THRESHOLD = dict(start=0.0, step=0.1)   # E = 0.5, H = 2 are MNE defaults


# --------------------------------------------------------------------------- #
# the published method, as a control
# --------------------------------------------------------------------------- #
def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH step-up, identical to `benjamini_hochberg_correction` in the
    published scripts (monotone adjusted p, capped at 1)."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 1.0
    for rank in range(n - 1, -1, -1):
        val = p[order[rank]] * n / (rank + 1)
        running = min(running, val)
        adj[order[rank]] = running
    return np.minimum(adj, 1.0)


def pointwise_t(A: np.ndarray, B: np.ndarray, alternative: str) -> Tuple[np.ndarray, np.ndarray]:
    """Paired t per window (A vs B), returning (t, p)."""
    n_time = A.shape[1]
    t = np.full(n_time, np.nan)
    p = np.full(n_time, np.nan)
    for k in range(n_time):
        res = stats.ttest_rel(A[:, k], B[:, k], alternative=alternative)
        t[k], p[k] = res.statistic, res.pvalue
    return t, p


def fdr_mask(A: np.ndarray, B: np.ndarray, alternative: str, alpha: float = ALPHA) -> Dict:
    t, p = pointwise_t(A, B, alternative)
    p_adj = benjamini_hochberg(p)
    return {"t": t, "p": p, "p_adj": p_adj, "sig": p_adj < alpha}


# --------------------------------------------------------------------------- #
# cluster permutation
# --------------------------------------------------------------------------- #
def _tail(alternative: str) -> int:
    return {"greater": 1, "less": -1, "two-sided": 0}[alternative]


def cluster_forming_threshold(n: int, alternative: str, alpha: float = ALPHA) -> float:
    """t at uncorrected p < alpha with df = n - 1 (the FieldTrip/MNE convention)."""
    df = n - 1
    if alternative == "two-sided":
        return float(stats.t.ppf(1 - alpha / 2, df))
    return float(stats.t.ppf(1 - alpha, df))


def cluster_test(A: np.ndarray, B: np.ndarray, alternative: str,
                 alpha: float = ALPHA, tfce: bool = False) -> Dict:
    """Sign-flip cluster permutation on the paired differences A - B.

    Returns per-window significance mask plus cluster-level detail.
    """
    D = A - B
    n = D.shape[0]
    tail = _tail(alternative)
    n_exact = 2 ** n                      # request more than exist -> MNE does all
    if tfce:
        threshold = TFCE_THRESHOLD
    else:
        thr = cluster_forming_threshold(n, alternative, alpha)
        threshold = -thr if tail == -1 else thr

    t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
        D, threshold=threshold, n_permutations=n_exact + 1, tail=tail,
        adjacency=None, out_type="mask", verbose=False,
    )

    n_time = D.shape[1]
    all_idx = np.arange(n_time)
    sig = np.zeros(n_time, dtype=bool)
    p_window = np.full(n_time, np.nan)
    detail: List[Dict] = []
    # MNE returns each 1-D cluster as a tuple holding a slice; index through it.
    if tfce:
        # one "cluster" per window: cluster_pv is already a per-window p.
        # t_obs here is the TFCE-transformed statistic, not t.
        for m, pv in zip(clusters, cluster_pv):
            p_window[m] = pv
        sig = p_window < alpha
        t_plain = stats.ttest_1samp(D, 0.0).statistic
        return {"t": t_plain, "tfce_score": t_obs, "p_window": p_window, "sig": sig,
                "clusters": detail, "threshold": threshold,
                "n_permutations": int(len(H0)), "n": n}
    for m, pv in zip(clusters, cluster_pv):
        idx = all_idx[m]
        p_window[idx] = np.minimum(np.nan_to_num(p_window[idx], nan=1.0), pv)
        detail.append({"windows": (int(idx[0]) + 1, int(idx[-1]) + 1),
                       "mass": float(t_obs[m].sum()), "p": float(pv)})
        if pv < alpha:
            sig[m] = True
    return {"t": t_obs, "p_window": p_window, "sig": sig, "clusters": detail,
            "threshold": threshold, "n_permutations": int(len(H0)), "n": n}


# --------------------------------------------------------------------------- #
# convenience
# --------------------------------------------------------------------------- #
def segments(sig: np.ndarray, x_grid: np.ndarray) -> List[Tuple[float, float]]:
    """Contiguous runs of True, as (first_x, last_x) -- same convention as the
    published `_compute_fdr_significant_segments`."""
    out: List[Tuple[float, float]] = []
    i = 0
    while i < len(sig):
        if sig[i]:
            j = i
            while j + 1 < len(sig) and sig[j + 1]:
                j += 1
            out.append((float(x_grid[i]), float(x_grid[j])))
            i = j
        i += 1
    return out


def _drop_nan_rows(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The published FDR helpers mask NaN window by window; a permutation test
    needs complete rows. Subjects with any missing window are dropped (and
    reported) rather than imputed."""
    keep = ~(np.isnan(A).any(axis=1) | np.isnan(B).any(axis=1))
    if not keep.all():
        print(f"cluster_stats: dropping {int((~keep).sum())} subject(s) with missing windows")
    return A[keep], B[keep]


def cluster_significant_segments(A: np.ndarray, B: np.ndarray, x_grid: np.ndarray,
                                 alpha: float = ALPHA, alternative: str = "two-sided"
                                 ) -> List[Tuple[float, float]]:
    """Drop-in replacement for the pipelines' `_compute_fdr_significant_segments`:
    contiguous x-intervals covered by significant clusters."""
    A, B = _drop_nan_rows(np.asarray(A, float), np.asarray(B, float))
    if A.shape[0] < 2:
        return []
    res = cluster_test(A, B, alternative, alpha)
    return segments(res["sig"], np.asarray(x_grid))


def cluster_results(A: np.ndarray, B: np.ndarray, x_grid: np.ndarray,
                    alpha: float = ALPHA, alternative: str = "two-sided") -> Dict:
    """Drop-in replacement for the pipelines' `_compute_fdr_results`: same keys
    ('segments', 'sig_mask', 'pvals', 'alpha') plus cluster detail. 'pvals' holds
    the cluster-level p of the cluster each window belongs to (NaN if none)."""
    A, B = _drop_nan_rows(np.asarray(A, float), np.asarray(B, float))
    x = np.asarray(x_grid)
    if A.shape[0] < 2:
        return {"alpha": alpha, "pvals": [], "pvals_adj": [], "sig_mask": [], "segments": [], "clusters": []}
    res = cluster_test(A, B, alternative, alpha)
    return {"alpha": alpha, "pvals": res["p_window"].tolist(), "pvals_adj": res["p_window"].tolist(),
            "sig_mask": res["sig"].tolist(), "segments": segments(res["sig"], x),
            "clusters": res["clusters"], "threshold": res["threshold"],
            "n_permutations": res["n_permutations"]}


def run_all(A: np.ndarray, B: np.ndarray, x_grid: np.ndarray, alternative: str,
            alpha: float = ALPHA) -> Dict:
    """FDR + fixed-threshold cluster + TFCE on the same matrices."""
    if np.isnan(A).any() or np.isnan(B).any():
        raise ValueError("NaN in input matrices; decide how to handle missing windows first")
    return {
        "fdr": fdr_mask(A, B, alternative, alpha),
        "cluster": cluster_test(A, B, alternative, alpha, tfce=False),
        "tfce": cluster_test(A, B, alternative, alpha, tfce=True),
        "x_grid": x_grid, "alternative": alternative, "n": A.shape[0],
    }
