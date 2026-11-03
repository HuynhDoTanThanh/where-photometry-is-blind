#!/usr/bin/env python3
"""The statistics of metric v1: pooled estimators from per-image counts, the paired image
bootstrap, the two-level (seed pair x image) bootstrap, the drive-clustered robustness
bootstrap, Holm, sigma_seed and the R2+R3 gain share. It computes; it never chooses.

How to use (library; tests/test_eval.py is the reference):
    import stats
    a = stats.counts(per_image_a, "all")          # {"n", "out", "epe_sum"} arrays, image order
    b = stats.counts(per_image_b, "all")
    stats.paired_image_bootstrap(a, b, stat="fl", B=10_000, seed=S)          # sensitivity row
    stats.two_level_bootstrap([m1, m2, m3], [pf1, pf2, pf3], stat="fl", B=10_000, seed=S)
    stats.holm([0.01, 0.04, 0.03, 0.005], alpha=0.05)  # -> [True, False, False, True]
B, the bootstrap seed, alpha and the multiplicity family come from prereg/PRIMARY.json (prereg
owns it); every result dict echoes them with the level and n_images, as each CI row must.
`python tools/eval/stats.py --selftest` runs a deterministic smoke and exits 0.

Rules (references/statistics.md)
    - The resampling unit is the image. Each replicate recomputes pooled Fl as
      100 * sum(out) / sum(n) over the drawn multiset; per-image Fl is never averaged.
    - Paired: ONE index draw per replicate, shared by both arms; seed i of M against seed i of PF.
      Both arms must have identical n per image (same GT): asserted before anything else.
    - The point estimate is the statistic on all images, never the bootstrap mean.
    - Two-level: per replicate draw 3 seed-pair indices, then one image draw shared across pairs
      and arms; report every per-seed delta and whether they share one sign.
    - Only numpy.random.default_rng(seed); no global seed, nothing from `random`.

Why: the first design averaged 3 seeds and then bootstrapped images, which treats seed variance
as zero (design review: statistical overclaiming). AUSE and rho_unc are per image (trap T28) and
never enter these estimators.
"""
from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence

import numpy as np

STATS = ("fl", "out3", "epe_pooled", "epe_img_mean")


class StatsError(ValueError):
    pass


def counts(per_image: Sequence[dict], region: str) -> dict[str, np.ndarray]:
    """Arrays over images, in order, from per-image records (regions.score_image 'r' blocks or
    rows of per_image.csv already parsed into the same shape)."""
    recs = [p["r"][region] for p in per_image]
    c = {"n": np.array([r["n"] for r in recs], np.int64),
         "out": np.array([r["out"] for r in recs], np.int64),
         "epe_sum": np.array([r["epe_sum"] for r in recs], np.float64)}
    if recs and "out3" in recs[0]:
        c["out3"] = np.array([r["out3"] for r in recs], np.int64)
    return c


def _check(c: dict) -> int:
    n = len(c["n"])
    for k, v in c.items():
        if len(v) != n:
            raise StatsError(f"count array {k} has {len(v)} images, n has {n}")
    if n == 0:
        raise StatsError("no images")
    return n


def statistic(c: dict, stat: str, idx: np.ndarray | None = None) -> float:
    """The split value of `stat` over the (multi)set idx of images (all images if None)."""
    n = c["n"] if idx is None else c["n"][idx]
    if stat in ("fl", "out3"):
        key = "out" if stat == "fl" else "out3"
        o = c[key] if idx is None else c[key][idx]
        tot = int(n.sum())
        return 100.0 * int(o.sum()) / tot if tot else math.nan
    e = c["epe_sum"] if idx is None else c["epe_sum"][idx]
    if stat == "epe_pooled":
        tot = int(n.sum())
        return float(e.sum()) / tot if tot else math.nan
    if stat == "epe_img_mean":
        keep = n > 0
        return float(np.mean(e[keep] / n[keep])) if keep.any() else math.nan
    raise StatsError(f"unknown statistic {stat!r} (one of {', '.join(STATS)})")


def _paired_ok(a: dict, b: dict) -> int:
    na, nb = _check(a), _check(b)
    if na != nb or not np.array_equal(a["n"], b["n"]):
        raise StatsError("the two arms do not share n per image: not scored against the same GT "
                         "and the same image order")
    return na


def _ci(d: np.ndarray, alpha: float) -> list[float]:
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [float(lo), float(hi)]


def paired_image_bootstrap(a: dict, b: dict, *, stat: str = "fl", B: int, seed: int,
                           alpha: float = 0.05) -> dict:
    """Delta = stat(a) - stat(b) with a percentile CI from B paired image resamples."""
    n = _paired_ok(a, b)
    rng = np.random.default_rng(seed)
    d = np.empty(B, np.float64)
    for i in range(B):
        idx = rng.integers(0, n, n)
        d[i] = statistic(a, stat, idx) - statistic(b, stat, idx)
    return {"level": "image", "stat": stat, "B": B, "seed": seed, "alpha": alpha,
            "n_images": n, "estimate": statistic(a, stat) - statistic(b, stat),
            "ci": _ci(d, alpha)}


def two_level_bootstrap(arm_m: Sequence[dict], arm_pf: Sequence[dict], *, stat: str = "fl",
                        B: int, seed: int, alpha: float = 0.05) -> dict:
    """Delta = mean_i [stat(M_i) - stat(PF_i)]: seed pairs resampled, then images (one image draw
    per replicate, shared by every pair and both arms)."""
    k = len(arm_m)
    if k == 0 or k != len(arm_pf):
        raise StatsError("two_level_bootstrap needs the same number (>= 1) of seeds per arm")
    n = _paired_ok(arm_m[0], arm_pf[0])
    for m, p in zip(arm_m, arm_pf, strict=True):
        if _paired_ok(m, p) != n or not np.array_equal(m["n"], arm_m[0]["n"]):
            raise StatsError("every seed must be scored on the same images and GT")
    per_seed = [statistic(m, stat) - statistic(p, stat) for m, p in zip(arm_m, arm_pf,
                                                                         strict=True)]
    rng = np.random.default_rng(seed)
    d = np.empty(B, np.float64)
    for i in range(B):
        s = rng.integers(0, k, k)
        idx = rng.integers(0, n, n)
        d[i] = float(np.mean([statistic(arm_m[j], stat, idx) - statistic(arm_pf[j], stat, idx)
                              for j in s]))
    signs = {math.copysign(1.0, x) for x in per_seed if x != 0}
    return {"level": "two-level", "stat": stat, "B": B, "seed": seed, "alpha": alpha,
            "n_images": n, "n_seed_pairs": k, "estimate": float(np.mean(per_seed)),
            "per_seed": per_seed, "all_negative": all(x < 0 for x in per_seed),
            "one_sign": len(signs) == 1 and 0.0 not in per_seed, "ci": _ci(d, alpha)}


def cluster_bootstrap(a: dict, b: dict, clusters: Sequence, *, stat: str = "fl", B: int,
                      seed: int, alpha: float = 0.05) -> dict:
    """Robustness only: resample clusters (e.g. raw drives) of images instead of images. The
    clustering rule for unmapped sequences is PRIMARY.json's; this function takes it as given."""
    n = _paired_ok(a, b)
    if len(clusters) != n:
        raise StatsError("one cluster label per image")
    labels = sorted(set(clusters), key=str)
    members = [np.array([i for i, c in enumerate(clusters) if c == lab], np.int64)
               for lab in labels]
    rng = np.random.default_rng(seed)
    d = np.empty(B, np.float64)
    for i in range(B):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[j] for j in pick])
        d[i] = statistic(a, stat, idx) - statistic(b, stat, idx)
    return {"level": "drive", "label": "robustness", "stat": stat, "B": B, "seed": seed,
            "alpha": alpha, "n_images": n, "n_clusters": len(members),
            "estimate": statistic(a, stat) - statistic(b, stat), "ci": _ci(d, alpha)}


def holm(pvals: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Holm step-down: sort, compare p_(i) with alpha / (m - i + 1), stop at the first failure.
    Returns the reject decisions in the input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    reject = [False] * m
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break
    return reject


def sigma_seed(values: Sequence[float]) -> float:
    """SD with ddof = 1 (PRIMARY.json sigma_seed_def: dev, pixel-pooled Fl-all, final step)."""
    if len(values) < 2:
        raise StatsError("sigma_seed needs at least 2 seeds")
    return float(np.std(np.asarray(values, np.float64), ddof=1))


def gain_share(pf_out_r23: int, m_out_r23: int, pf_out_all: int, m_out_all: int) -> float | str:
    """Share of the outlier reduction that falls on R2+R3, from pooled counts. 'undefined' when
    the total reduction is <= 0 (condition (a) fails then anyway)."""
    total = pf_out_all - m_out_all
    if total <= 0:
        return "undefined"
    return (pf_out_r23 - m_out_r23) / total


def _selftest() -> int:
    rng = np.random.default_rng(0)
    n = rng.integers(1000, 5000, 50)
    a = {"n": n, "out": (n * 0.06).astype(np.int64), "epe_sum": n * 1.8}
    b = {"n": n, "out": (n * 0.07).astype(np.int64), "epe_sum": n * 1.9}
    r1 = paired_image_bootstrap(a, b, B=500, seed=7)
    r2 = paired_image_bootstrap(a, b, B=500, seed=7)
    assert r1 == r2 and r1["ci"][0] <= r1["estimate"] <= r1["ci"][1]
    assert holm([0.01, 0.04, 0.03, 0.005]) == [True, False, False, True]
    print("STATS_SELFTEST_OK")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    sys.exit(_selftest() if ap.parse_args().selftest else 64)
