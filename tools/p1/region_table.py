#!/usr/bin/env python3
"""region_table.py: the P1 region table (plan row W3-02, cycle CYC-003) built from CACHED
predictions only: per source x pose x window x region, EPE, Fl and coverage with image-level
bootstrap CIs, on the dev half of KITTI-15 and on KITTI-12. CPU, on the Mac. It computes numbers;
it writes no ledger row (hard rule 7) and starts no VM.

What it does.
    For every entry of a spec file (one source x pose x window, pointing at a cache of flows at the
    frame's native resolution) it reads each image's flow, takes the region masks of ONE provider
    (tools/eval/regions.py by default; the R4 split is staged, see below) and the GT of the Mac GT
    root, and keeps per image and region integer counts:
        n        region pixels (GT-valid, from regions.py)
        n_cov    of those, pixels where the source has a finite flow (NaN = no teacher depth)
        out_cov  outliers among the covered pixels (regions.outliers: EPE > 3 and EPE/|F| > 0.05)
        out_fc   fail-closed outliers: out_cov + (n - n_cov)
        epe_sum  sum of EPE over the covered pixels (float64)
    Per cell (source, pose, window, dataset, region) the table reports
        coverage            100 * sum(n_cov) / sum(n)                        (px-pooled)
        Fl fail-closed      100 * sum(out_fc) / sum(n)    an uncovered pixel is an outlier, so a
                            sparse teacher cannot win by abstaining [E prereg/drafts/GATE-1.json
                            _sources.teacher_coverage; a DRAFT convention, the user commits it]
        Fl covered          100 * sum(out_cov) / sum(n_cov)                  (px-pooled)
        EPE covered         pooled (sum epe / sum n_cov) AND image-mean (images with n_cov > 0),
                            both named: an EPE without its aggregation is never quoted (metric s8)
        Out3 fail-closed    KITTI-12 only, the official Out-All rule beside Fl(K15-rule)
    each with its percentile CI, plus n_images (images with n > 0 in the region), n_pixels (sum n)
    and n_images_scored (all images of the cell). Coverage is reported beside Fl, never folded in.
    Dataset names: kitti15 = the dev half (100 ids of splits/k15_split_v1.json); kitti12 = ids
    0-193, whose regions are the K12-R1..3 of regions.py (R1 = K12-R1 ...), with R3-rel and the R4
    family 'n/a' (no obj_map ships with KITTI-12; R3-rel has no K12 definition in regions.py).

Statuses of a cell (never a silent 0.0)
    ok        numbers present
    n=0       no GT pixel of this region in any image: value null, n_images 0, n_pixels 0 (the
              evaluator's zero-append trap, metric skill s2 item 2: a region EPE counts only images
              with n_R > 0)
    staged    the region provider cannot make this region yet (R4m, R4s, R4m-epi live in
              tools/staging/eval/regions_r4.py until its regions.py patch lands). Nothing is
              faked: the column carries the reason
    n/a       the region does not exist for this dataset
    partial   numbers present, but fewer images than the split holds (100 dev, 194 KITTI-12): the
              cell carries complete=false and n_expected, the Markdown says PARTIAL k/N; such
              rows are not comparable across sources
    withheld-S6  R2 while regions.R2_NEAR_EMPTY is None: regions.md S6 fixes the 'near-empty'
              threshold BEFORE any R2 count is seen; --acknowledge-s6 --s6-reason TEXT lifts it,
              and every R2 cell and the Markdown then carry an S6 caveat (exploratory)
    missing   the source's cache is not on disk, or yielded no image (e.g. COLMAP): the cell is
              listed, exit 0 unless --strict, which also fails on partial sources (exit 2)
Statistics (tools/eval/stats.py, metric skill s5): the resampling unit is the image; every
replicate draws images with replacement and recomputes the pooled statistic from the counts
(stats.statistic; a per-image Fl is never averaged). B, level and the bootstrap seed come from
prereg/PRIMARY.json primary_endpoint.ci; its bootstrap_seed is null until prereg-v3, so --seed is
needed (recorded as a diagnostic seed). One draw of image indices per cell is shared by all of its
regions and statistics.

How to invoke (repo root; the venv of the Mac tools):
    .venv/bin/python tools/p1/region_table.py --spec SPEC.json --seed 1 \
        [--regions v1|staged-r4|module:PATH.py] [--gt-root /Volumes/Shin/U2FLOW_data/gt] \
        [--acknowledge-s6 --s6-reason TEXT] [--B 10000] [--alpha 0.05] [--strict] \
        [--json-out reports/p1_region/region_table.json] \
        [--md-out reports/p1_region/region_table.md]
    Default output: the JSON on stdout (--stdout-format md for the Markdown). An output path is an
    ALLOWLIST: only a new file (or one this tool wrote) under <repo>/reports/p1_region/, compared
    case-insensitively on the resolved path; trackers, generated views, ledgers, an existing file
    of anyone else and a symlink are exit 64. An empty spec is exit 64.
    SPEC.json: {"schema": 1, "sources": [ENTRY, ...]}; ENTRY keys
        source    mapanything | pi3 | colmap | selftri | monodepth2 | raftct | u2flow | u2flow_ff
        pose      native | ref | n/a          window   w03-11 | w10-18 | n/a
        role      teacher (oracle=true) | student
        format    export  a render_frig.py export-preds out dir (<ds>/preds + COVERAGE.json)
                  preds   an eval_kitti PREDS.json dir; needs "dataset": kitti15|kitti12
                  run     an eval_kitti.py run dir (metrics.json + preds/); split dev|k12 only
                  cache   a p1lib/cache.py cache dir (diag caches), needs "flow_key"
                          (native_12 | native_10to11 | native | ...)
        path      the directory
        optional  datasets (list), protocol, flags, packing, gpu_name, lock_hash (provenance the
                  cache itself does not record)
    tools/p1/specs/ may hold such files; none is shipped with real paths until the caches exist.
Output: the JSON {schema, header, cells} and the Markdown rendering. The header records the
protocol key fields (metric skill s7: split, protocol, eval_sha, gt_pack_sha, agg, flags, oracle,
gpu_name, lock_hash, packing), per source and for the table, and the statistics settings.
Exit codes: 0 ok · 1 error (a sanity check S1-S3 failed, a shape differs, bad spec) · 2 precondition
(a source is missing under --strict, GT root absent) · 3 refused (--stdout and --*-out collide) ·
7 integrity (a shard or GT file does not match its sha256) · 50 seal (a held-out or all-200 input,
a path under the seal; a STOP) · 64 usage.

How the numbers could enter the ledger (this tool never does it): rows are written only by
note.py and score_cached.py. A teacher export has no metrics.json, so score_cached cannot take it
until W3-03 gives it the coverage rule; until then the JSON is a measured artefact to record with
`note.py evidence --kind measurement --standard measured --source tools/p1/region_table.py
--data '{"artifact": PATH, "artifact_sha256": SHA}'`. Teacher rows are oracle=true rows.

Why it is built this way.
- One kernel: masks, the EPE map and the outlier rule come from tools/eval/regions.py, so a teacher
  and the student share one code path (metric s4); this file adds only the NaN (coverage) rule and
  the bookkeeping. tools/eval/* is never edited.
- Dev and KITTI-12 only, refused loudly: a held-out id in any input, a split other than dev/k12 and
  any path under the seal are exit 50, never a skipped row (hard rule 4). The GT root is the Mac's
  verified dev GT, which holds no held-out file anyway.
- Region EPE over images with n_R > 0 and an explicit n: U2Flow's own noc/occ EPE appends 0.0 for
  empty images and pulls the mean down (evaluate.py:559-562, trap T5).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "tools" / "eval"))

import eval_kitti as ek  # noqa: E402
import regions as rg  # noqa: E402
import stats as st  # noqa: E402
from p1lib import cache as CA  # noqa: E402
from p1lib import common as C  # noqa: E402
from p1lib import kitti as KT  # noqa: E402

EXIT_OK, EXIT_ERR, EXIT_PRE, EXIT_REFUSED, EXIT_INTEGRITY, EXIT_SEAL, EXIT_USAGE = (
    0, 1, 2, 3, 7, 50, 64)
SCHEMA = 1
REGIONS = ("R1", "R2", "R3", "R3-rel", "R4m", "R4s", "R4m-epi")
K12_KERNEL = {"R1": "K12-R1", "R2": "K12-R2", "R3": "K12-R3"}
EXPECTED_SOURCES = ("mapanything", "pi3", "colmap", "selftri", "monodepth2", "raftct", "u2flow",
                    "u2flow_ff")
POSES = ("native", "ref", "n/a")
WINDOWS = ("w03-11", "w10-18", "n/a")
FORMATS = ("export", "preds", "run", "cache")
DATASETS = ("kitti15", "kitti12")
SPLIT_OF = {"kitti15": "dev", "kitti12": "k12"}
EXPECT_N = {"kitti15": 100, "kitti12": len(ek.K12_IDS)}
STAGED_R4 = REPO / "tools" / "staging" / "eval" / "regions_r4.py"
OUT_DIR = "reports/p1_region"        # the only place an output may go (allowlist, see out_path)
MD_MARK = "# P1 region table"
STATS_ORDER = ("coverage", "fl_failclosed", "fl_covered", "epe_covered_pooled",
               "epe_covered_img_mean")


class Fail(Exception):
    def __init__(self, code: int, msg: str) -> None:
        super().__init__(msg)
        self.code = code


# ------------------------------------------------------------------------------ region providers
class V1Provider:
    """regions.py as it is: R1, R2, R3, R3-rel for KITTI-15 (K12-R1..3 for KITTI-12). The R4
    split is not here, so R4m, R4s and R4m-epi are 'staged'."""

    name = "regions_v1 (tools/eval/regions.py)"
    version = rg.REGIONS_VERSION
    staged_reason = ("R4m, R4s, R4m-epi need the regions_v2 R4 split, staged in "
                     "tools/staging/eval/regions_r4.py until its regions.py patch lands")

    def masks(self, src, img_id: str, gt, dataset: str) -> dict:
        if dataset == "kitti12":
            # [M 2026-10-06] on KITTI-12, flow_noc differs from flow_occ on V_noc in all 194
            # images (2.6% of pixels, up to 6.8 px), so regions.py's S3 (a KITTI-15 property)
            # always fails. The masks use valid_noc only and the noc selection is scored against
            # flow_occ (evaluate.py convention), so S3 is made vacuous here, not weakened.
            gt = rg.GTImage(gt.flow_occ, gt.valid_occ, gt.flow_occ, gt.valid_noc, gt.obj_map)
        try:
            return rg.masks(gt, dataset)[0]
        except rg.RegionError as e:
            raise Fail(EXIT_ERR, f"{dataset} {img_id}: regions sanity check failed: {e}") from None

    def close(self) -> None:
        pass


class StagedR4Provider(V1Provider):
    """regions.py plus the staged regions_r4.split_r4 (dev GT only: disp_occ_0 and the calibration
    of u2f-meta-v1), loaded from tools/staging/eval/regions_r4.py. Never edits or lands it."""

    name = "regions_v2 draft (tools/eval/regions.py + staged tools/staging/eval/regions_r4.py)"
    version = "regions_v2-staged"
    staged_reason = ""

    def __init__(self, gt_root: Path | None, r4_path: Path = STAGED_R4) -> None:
        if not r4_path.is_file():
            raise Fail(EXIT_PRE, f"{r4_path} not found")
        spec = importlib.util.spec_from_file_location("u2f_region_table_r4", r4_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        self.r4, self.path = mod, r4_path
        self.calib = mod.find_calib(gt_root)
        if self.calib is None:
            raise Fail(EXIT_PRE, "no calibration source for the R4 split (u2f-meta-v1 zip beside "
                       "the GT root, or $U2_K15_CALIB)")

    def masks(self, src, img_id: str, gt, dataset: str) -> dict:
        m = super().masks(src, img_id, gt, dataset)
        if dataset != "kitti15":
            return m
        r4 = self.r4
        try:
            disp, dvalid = r4.decode_disp(src.read_bytes("disp_occ_0", img_id))
            masks, _ = r4.split_r4(gt.flow_occ, gt.valid_occ >= 0.5, gt.valid_noc >= 0.5,
                                   gt.obj_map, disp, dvalid, self.calib.read(img_id))
        except r4.R4Error as e:
            raise Fail(EXIT_ERR, f"{img_id}: R4 sanity check failed: {e}") from None
        return {**m, **masks}

    def close(self) -> None:
        self.calib.close()


def module_provider(path: Path, gt_root: Path | None):
    """A provider from a file: `make_provider(gt_root)` returns an object with
    `masks(src, img_id, gt, dataset) -> {region: bool mask}`, `name`, `version`, optionally
    `staged_reason` (non-empty = the R4 columns are staged) and `close()`."""
    if not path.is_file():
        raise Fail(EXIT_PRE, f"{path} not found")
    spec = importlib.util.spec_from_file_location("u2f_region_provider", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    prov = mod.make_provider(gt_root)
    if not hasattr(prov, "staged_reason"):
        prov.staged_reason = ""
    if not hasattr(prov, "close"):
        prov.close = lambda: None
    return prov


def make_provider(kind: str, gt_root: Path | None):
    """('v1' | 'staged-r4' | 'module:PATH'). An unavailable staged provider degrades to v1 with the
    reason, so the R4 columns say 'staged' instead of the run failing."""
    if kind == "v1":
        return V1Provider()
    if kind == "staged-r4":
        try:
            return StagedR4Provider(gt_root)
        except Fail as e:
            prov = V1Provider()
            prov.staged_reason = f"the staged R4 provider is unavailable: {e}"
            return prov
    if kind.startswith("module:"):
        return module_provider(Path(kind[7:]), gt_root)
    raise Fail(EXIT_USAGE, f"--regions {kind!r}: v1, staged-r4 or module:PATH")


# ------------------------------------------------------------------------------ GT and masks
class GTCache:
    """A bounded cache of per-image GT (flow_occ and the region masks, bit-packed): the masks
    depend on GT only, so every source of a dataset reuses them. limit 0 disables it."""

    def __init__(self, src, provider, dataset: str, limit: int) -> None:
        self.src, self.provider, self.dataset, self.limit = src, provider, dataset, limit
        self.lru: OrderedDict = OrderedDict()

    def get(self, img_id: str) -> tuple:
        if img_id in self.lru:
            self.lru.move_to_end(img_id)
            return self.lru[img_id]
        gt = self.src.read(img_id, need_obj=self.dataset == "kitti15")
        masks = self.provider.masks(self.src, img_id, gt, self.dataset)
        ent = (gt.shape, gt.flow_occ, {k: np.packbits(v.reshape(-1)) for k, v in masks.items()})
        if self.limit > 0:
            self.lru[img_id] = ent
            while len(self.lru) > self.limit:
                self.lru.popitem(last=False)
        return ent


def unpack(packed: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.unpackbits(packed)[: shape[0] * shape[1]].reshape(shape).astype(bool)


def kernel_name(region: str, dataset: str) -> str | None:
    """The regions.py mask behind a table region for a dataset; None = not defined there."""
    if dataset == "kitti15":
        return region
    return K12_KERNEL.get(region)


# ------------------------------------------------------------------------------ per-image counts
def pixel_maps(flow: np.ndarray, shape, flow_occ: np.ndarray, dataset: str) -> tuple:
    """(finite, epe, out) (H, W) maps of one prediction: finite = every flow component is finite
    (else 'uncovered'), epe and out from regions.py's kernel on the finite-zeroed flow (out is
    only meaningful where finite). Shared by image_counts and tools/p1/composite_bound.py."""
    flow = np.asarray(flow, np.float32)
    if flow.shape != (2, *shape):
        raise Fail(EXIT_ERR, f"prediction shape {flow.shape} != GT (2, {shape[0]}, {shape[1]}): "
                   "only flows at the frame's native resolution are scored")
    finite = np.isfinite(flow).all(axis=0)
    safe = np.where(finite[None], flow, np.float32(0.0))
    epe = rg.epe_map(safe, flow_occ)
    out = rg.outliers(epe, rg.magnitude(flow_occ), eps=dataset == "kitti15")
    return finite, epe, out


def image_counts(flow: np.ndarray, shape, flow_occ: np.ndarray, masks: dict, dataset: str,
                 regions: list[str]) -> dict:
    """{region: {n, n_cov, out_cov, out_fc, epe_sum[, out3_fc]}} for one image. A non-finite
    flow component is 'uncovered' (no teacher depth); everything else is regions.py's kernel."""
    finite, epe, out = pixel_maps(flow, shape, flow_occ, dataset)
    out3 = epe > np.float32(3.0)
    res = {}
    for r in regions:
        m = masks[r]
        cov = m & finite
        n, n_cov = int(m.sum()), int(cov.sum())
        rec = {"n": n, "n_cov": n_cov, "out_cov": int((out & cov).sum()),
               "epe_sum": float(epe[cov].sum(dtype=np.float64))}
        rec["out_fc"] = rec["out_cov"] + (n - n_cov)
        if dataset == "kitti12":
            rec["out3_fc"] = int((out3 & cov).sum()) + (n - n_cov)
        res[r] = rec
    return res


# ------------------------------------------------------------------------------ statistics
def _pack(vals: list, key: str) -> np.ndarray:
    return np.array([v[key] for v in vals], np.float64 if key == "epe_sum" else np.int64)


def region_arrays(per_image: list[dict], region: str) -> dict:
    recs = [p[region] for p in per_image]
    return {k: _pack(recs, k) for k in recs[0]}


def stat_sets(a: dict, k12: bool) -> dict:
    """The count arrays behind each reported statistic, in stats.py's own shape. Coverage reuses
    the 'fl' estimator with the covered count in the outlier slot (100 * sum / sum)."""
    sets = {
        "coverage": ({"n": a["n"], "out": a["n_cov"]}, "fl"),
        "fl_failclosed": ({"n": a["n"], "out": a["out_fc"]}, "fl"),
        "fl_covered": ({"n": a["n_cov"], "out": a["out_cov"]}, "fl"),
        "epe_covered_pooled": ({"n": a["n_cov"], "epe_sum": a["epe_sum"]}, "epe_pooled"),
        "epe_covered_img_mean": ({"n": a["n_cov"], "epe_sum": a["epe_sum"]}, "epe_img_mean"),
    }
    if k12:
        sets["out3_failclosed"] = ({"n": a["n"], "out": a["out_fc"], "out3": a["out3_fc"]},
                                   "out3")
    return sets


AGG = {"coverage": "px-pooled", "fl_failclosed": "px-pooled", "fl_covered": "px-pooled",
       "epe_covered_pooled": "px-pooled", "epe_covered_img_mean": "img-mean",
       "out3_failclosed": "px-pooled"}


def _num(x) -> float | None:
    x = float(x)
    return None if x != x else x


def cell_stats(per_image: list[dict], regions: list[str], dataset: str, boot: dict) -> dict:
    """{region: stats dict} for one cell. One image-index draw (B x N) serves every region and
    statistic, so the CIs of one cell are mutually consistent."""
    n_img = len(per_image)
    draws = None
    if boot["B"] > 0:
        draws = np.random.default_rng(boot["seed"]).integers(0, n_img, (boot["B"], n_img))
    lo_p, hi_p = 100 * boot["alpha"] / 2, 100 * (1 - boot["alpha"] / 2)
    out = {}
    for r in regions:
        a = region_arrays(per_image, r)
        n_px = int(a["n"].sum())
        rec = {"n_images_scored": n_img, "n_images": int((a["n"] > 0).sum()), "n_pixels": n_px,
               "n_covered": int(a["n_cov"].sum())}
        if n_px == 0:
            rec.update(status="n=0", reason="no GT pixel of this region in any scored image")
            out[r] = rec
            continue
        rec["status"] = "ok"
        for name, (counts, stat) in stat_sets(a, dataset == "kitti12").items():
            point = _num(st.statistic(counts, stat))
            if name == "epe_covered_img_mean":
                rec["n_images_epe"] = int((a["n_cov"] > 0).sum())
            entry = {"value": point, "agg": AGG[name], "ci": None}
            if point is not None and draws is not None:
                reps = np.array([st.statistic(counts, stat, idx) for idx in draws], np.float64)
                if np.isnan(reps).any():
                    entry["ci_nan_replicates"] = int(np.isnan(reps).sum())
                reps = reps[~np.isnan(reps)]
                if len(reps):
                    lo, hi = np.percentile(reps, [lo_p, hi_p])
                    entry["ci"] = [float(lo), float(hi)]
            if point is None:
                entry["note"] = "no covered pixel: undefined, not 0.0"
            rec[name] = entry
        out[r] = rec
    return out


# ------------------------------------------------------------------------------ sources
def _dev_ids(root: Path) -> tuple[set[str], set[str]]:
    dev, held = ek.split_ids(root)
    return set(dev), set(held)


def _check_ids(dataset: str, ids, dev: set, held: set, where: str) -> None:
    """The seal, before a single flow is read: no KITTI-15 id outside the dev half."""
    ids = list(ids)
    if len(set(ids)) != len(ids):
        raise Fail(EXIT_INTEGRITY, f"{where}: an image id occurs twice")
    if dataset == "kitti15":
        bad = [i for i in ids if i not in dev]
        if bad:
            raise Fail(EXIT_SEAL, f"{where}: {len(bad)} KITTI-15 id(s) outside the dev half "
                       f"({'held-out' if any(i in held for i in bad) else 'unknown'}, e.g. "
                       f"{bad[0]}): held-out and all-200 are never an input (hard rule 4)")
    else:
        bad = [i for i in ids if i not in ek.K12_IDS]
        if bad:
            raise Fail(EXIT_ERR, f"{where}: KITTI-12 ids outside 000000-000193: {bad[:3]}")


def _stream_preds(pred_dir: Path, dataset: str, dev: set, held: set):
    C.refuse_sealed(pred_dir)              # a run's 'preds' key may point anywhere
    doc = json.loads((pred_dir / "PREDS.json").read_text("utf-8"))
    ids = [i for sh in doc["shards"] for i in sh["ids"]]
    _check_ids(dataset, ids, dev, held, str(pred_dir))
    shards = doc["shards"]

    def it():
        try:
            yield from sorted(ek.read_preds(pred_dir), key=lambda t: t[0])
        except rg.IntegrityError as e:
            raise Fail(EXIT_INTEGRITY, str(e)) from None

    return it, ek.pred_sha(shards), len(ids)


def _stream_cache(idx: CA.CacheIndex, key: str, dataset: str, win: str, dev: set, held: set):
    parsed = {}
    for item in idx.ids():
        d = KT.parse_id(item)
        if d["ds"] == dataset and d["sub"] == "training" and d["side"] == "image_2":
            parsed[item] = d
    for item, d in parsed.items():
        if "win" in d and d["win"] != win:
            raise Fail(EXIT_ERR, f"{item}: the cache holds window {d['win']} but the spec says "
                       f"{win}: a mislabelled window is refused")

    # The seal first, before any image is scored: a non-dev KITTI-15 item may exist in a cache
    # (it is a window of the whole split), but never a flow of it.
    if dataset == "kitti15":
        for item, d in parsed.items():
            if d["seq"] not in dev:
                got = idx.get(item)
                if got is not None and key in got[0]:
                    raise Fail(EXIT_SEAL, f"{item}: a prediction ({key}) of a non-dev KITTI-15 "
                               "id (hard rule 4)")

    def it():
        seen_keys: set = set()
        n_loaded = n_yield = 0
        for item in sorted(parsed, key=lambda i: parsed[i]["seq"]):
            seq = parsed[item]["seq"]
            if dataset == "kitti15" and seq not in dev:
                continue                          # checked above: no flow, nothing to score
            got = idx.get(item)
            if got is None:
                continue
            n_loaded += 1
            seen_keys |= set(got[0])
            if key not in got[0]:
                continue
            n_yield += 1
            yield seq, np.asarray(got[0][key], np.float32)
        if n_loaded and not n_yield:
            raise Fail(EXIT_ERR, f"flow_key {key!r} is in no {dataset} item of the cache; the "
                       f"keys that exist: {sorted(seen_keys)}")

    shas = sorted(f"{k}:{v}" for k, v in idx.sha.items())
    return it, rg.sha256_bytes("\n".join(shas).encode()), None


def open_entry(entry: dict, root: Path, dev: set, held: set) -> tuple[dict, dict]:
    """(provenance, {dataset: (iterator factory, pred_sha, n_declared)}) for one spec entry, with
    every seal check done before any pixel is read. A missing path raises Fail(EXIT_PRE)."""
    path = Path(entry["path"])
    try:
        C.refuse_sealed(path)
        return _open_entry(entry, root, dev, held)
    except C.Stop as e:
        raise Fail(e.code, e.msg) from None


def _open_entry(entry: dict, root: Path, dev: set, held: set) -> tuple[dict, dict]:
    fmt, path = entry["format"], Path(entry["path"])
    if not path.exists():
        raise Fail(EXIT_PRE, f"{path}: not on disk")
    want = list(entry.get("datasets") or DATASETS)
    prov = {k: entry.get(k) for k in ("protocol", "flags", "packing", "gpu_name", "lock_hash")}
    streams: dict = {}
    if fmt == "export":
        cov_p = path / "COVERAGE.json"
        if cov_p.is_file():
            cov = json.loads(cov_p.read_text("utf-8"))
            env = cov.get("env") or {}
            prov.update(gpu_name=prov["gpu_name"] or env.get("gpu_name"),
                        gpu_source=env.get("gpu_source"), render_host=env.get("render_host"),
                        pose_cv2=env.get("pose_cv2"), k=env.get("k"), k_source=env.get("k_source"),
                        row_kind=cov.get("row_kind"), flow_key=cov.get("flow_key"))
        for ds in want:
            if (path / ds / "preds" / "PREDS.json").is_file():
                streams[ds] = _stream_preds(path / ds / "preds", ds, dev, held)
    elif fmt == "preds":
        ds = entry.get("dataset")
        if ds not in DATASETS:
            raise Fail(EXIT_USAGE, f"{path}: format preds needs dataset kitti15|kitti12")
        streams[ds] = _stream_preds(path, ds, dev, held)
    elif fmt == "run":
        m = json.loads((path / "metrics.json").read_text("utf-8"))
        if m.get("split") not in ("dev", "k12"):
            raise Fail(EXIT_SEAL, f"{path}: run split {m.get('split')!r}: only dev and k12 runs "
                       "are an input (held-out and all-200 never)")
        ds = ek.SPLITS[m["split"]]
        env = m.get("env") or {}
        prov.update(gpu_name=prov["gpu_name"] or env.get("gpu_name"),
                    lock_hash=prov["lock_hash"] or env.get("lock_hash"),
                    ckpt_sha=m.get("ckpt_sha"), ff=m.get("ff"), producer_eval_sha=m.get("eval_sha"),
                    producer_git_sha=m.get("git_sha"), producer_partial=m.get("partial"))
        streams[ds] = _stream_preds(path / m.get("preds", "preds"), ds, dev, held)
    elif fmt == "cache":
        key = entry.get("flow_key")
        if not key:
            raise Fail(EXIT_USAGE, f"{path}: format cache needs flow_key")
        idx = CA.CacheIndex(path)
        prov["flow_key"] = key
        for ds in want:
            streams[ds] = _stream_cache(idx, key, ds, entry["window"], dev, held)
    return prov, streams


def check_entry(e: dict) -> None:
    for k in ("source", "pose", "window", "role", "format", "path"):
        if k not in e:
            raise Fail(EXIT_USAGE, f"spec entry {e.get('source')!r} lacks {k!r}")
    if e["pose"] not in POSES or e["window"] not in WINDOWS or e["format"] not in FORMATS:
        raise Fail(EXIT_USAGE, f"spec entry {e['source']}: pose {POSES}, window {WINDOWS}, "
                   f"format {FORMATS}")
    if e["role"] not in ("teacher", "student"):
        raise Fail(EXIT_USAGE, f"spec entry {e['source']}: role teacher|student")
    for k in ("split", "splits"):
        v = e.get(k)
        if v is not None and not set([v] if isinstance(v, str) else v) <= {"dev", "k12"}:
            raise Fail(EXIT_SEAL, f"spec entry {e['source']}: {k}={v!r}: only dev and k12 exist "
                       "here (held-out, full200 and all-200 are never an input)")


# ------------------------------------------------------------------------------ the table
def build_table(spec: dict, *, root: Path, gt_open, provider, boot: dict,
                acknowledge_s6: bool = False, gt_cache_images: int = 300,
                gt_root: Path | None = None, expect_n: dict | None = None) -> dict:
    """The table document. gt_open(dataset) -> a GT source with read(id, need_obj), read_bytes,
    close (regions.GTSource, or a stand-in in tests). Raises Fail on a seal, integrity or spec
    problem; a source whose path is missing becomes 'missing' cells."""
    entries = list(spec.get("sources") or [])
    if not entries:
        raise Fail(EXIT_USAGE, "the spec lists no sources (a mistyped key?): an empty table is "
                   "never a result")
    expect = {**EXPECT_N, **(expect_n or {})}
    for e in entries:
        check_entry(e)
    dev, held = _dev_ids(root)
    gts, caches = {}, {}
    cells, prov_list, absent = [], [], []
    try:
        for e in entries:
            label = {"source": e["source"], "pose": e["pose"], "window": e["window"]}
            oracle = e["role"] == "teacher"
            try:
                prov, streams = open_entry(e, root, dev, held)
            except Fail as ex:
                if ex.code != EXIT_PRE:
                    raise
                prov_list.append({**label, "role": e["role"], "oracle": oracle, "status": "missing",
                                  "path": str(e["path"]), "why": str(ex), "complete": False})
                for ds in _wanted(e):
                    cells += [_absent(label, ds, r, provider, str(ex))
                              for r in _all_regions(provider, ds)]
                continue
            prov_row = {**label, "role": e["role"], "oracle": oracle, "path": str(e["path"]),
                        "format": e["format"], "status": "present", **prov, "datasets": {}}
            for ds in (list(streams) if e["format"] == "run" else _wanted(e)):
                if ds not in streams:
                    prov_row["datasets"][ds] = {"status": "not in this source",
                                                "n_expected": expect[ds], "complete": False}
                    cells += [_absent(label, ds, r, provider, "this source holds no " + ds)
                              for r in _all_regions(provider, ds)]
                    continue
                factory, pred_sha, _n = streams[ds]
                if ds not in gts:
                    gts[ds] = gt_open(ds)
                    caches[ds] = GTCache(gts[ds], provider, ds, gt_cache_images)
                regs = [r for r in _all_regions(provider, ds) if _available(r, ds, provider)]
                per_image, ids = [], []
                for seq, flow in factory():
                    shape, flow_occ, packed = caches[ds].get(seq)
                    masks = {k: unpack(v, shape) for k, v in packed.items()}
                    per_image.append(image_counts(flow, shape, flow_occ, masks, ds,
                                                  [kernel_name(r, ds) for r in regs]))
                    ids.append(seq)
                complete = len(ids) == expect[ds]
                prov_row["datasets"][ds] = {"status": "scored" if ids else "empty",
                                            "pred_sha": pred_sha, "n_images": len(ids),
                                            "n_expected": expect[ds], "complete": complete}
                # re-key the per-image counts by table region
                keyed = [{r: p[kernel_name(r, ds)] for r in regs} for p in per_image]
                stats_by_region = cell_stats(keyed, regs, ds, boot) if keyed else {}
                for r in _all_regions(provider, ds):
                    cells.append(_cell(label, ds, r, stats_by_region, provider, acknowledge_s6,
                                       bool(keyed), complete, expect[ds]))
            dss = prov_row["datasets"].values()
            if not any(d.get("n_images") for d in dss):
                prov_row["status"] = "missing"
                prov_row["why"] = "no image of this source was scored (empty or wrong key)"
            prov_row["complete"] = bool(dss) and all(d["complete"] for d in dss)
            prov_list.append(prov_row)
    finally:
        for g in gts.values():
            g.close()
    present = {(p["source"]) for p in prov_list}
    absent = [s for s in EXPECTED_SOURCES if s not in present]
    return {"schema": SCHEMA, "header": _header(root, prov_list, absent, provider, boot,
                                                acknowledge_s6, gt_root), "cells": cells}


def _wanted(e: dict) -> list[str]:
    """The datasets a spec entry is expected to cover (a preds dir holds exactly one)."""
    if e["format"] == "preds":
        return [e["dataset"]] if e.get("dataset") in DATASETS else list(DATASETS)
    if e["format"] == "run":
        return list(DATASETS)         # narrowed to the run's own dataset once it is opened
    return list(e.get("datasets") or DATASETS)


def _absent(label: dict, ds: str, r: str, provider, reason: str) -> dict:
    """A cell with no data behind it: 'n/a' / 'staged' where the region itself says so, else
    'missing'."""
    why = _why_not(r, ds, provider)
    return _blank(label, ds, r, *(why or ("missing", reason)))


def _all_regions(provider, ds: str) -> list[str]:
    """The table's regions; R4u (the undetermined share of R4) only when the provider makes it."""
    return list(REGIONS) + (["R4u"] if ds == "kitti15" and not provider.staged_reason else [])


def _available(r: str, ds: str, provider) -> bool:
    return _why_not(r, ds, provider) is None


def _why_not(r: str, ds: str, provider) -> tuple[str, str] | None:
    """('n/a' | 'staged', reason) when the table region cannot be computed, else None."""
    if kernel_name(r, ds) is None:
        return "n/a", ("R3-rel has no KITTI-12 definition in regions.py" if r == "R3-rel"
                       else "KITTI-12 ships no obj_map: no R4 family")
    if ds == "kitti15" and r in ("R4m", "R4s", "R4m-epi", "R4u") and provider.staged_reason:
        return "staged", provider.staged_reason
    return None


def _blank(label: dict, ds: str, r: str, status: str, reason: str) -> dict:
    return {**label, "dataset": ds, "split": SPLIT_OF[ds], "region": r,
            "kernel_region": kernel_name(r, ds), "status": status, "reason": reason,
            "n_images": None, "n_pixels": None}


S6_CAVEAT = ("S6: regions.R2_NEAR_EMPTY is not fixed (regions.md S6); this R2 number was seen "
             "before the threshold: exploratory, not for a claim")


def _cell(label, ds, r, stats_by_region, provider, ack, have_images, complete=True,
          n_expected=None) -> dict:
    why = _why_not(r, ds, provider)
    if why:
        return _blank(label, ds, r, *why)
    if r == "R2" and rg.R2_NEAR_EMPTY is None and not ack:
        return _blank(label, ds, r, "withheld-S6", "regions.md S6: the R2 near-empty threshold is "
                      "not fixed (regions.R2_NEAR_EMPTY is None); no R2 count is shown until it "
                      "is, or --acknowledge-s6 with --s6-reason")
    if not have_images:
        return _blank(label, ds, r, "missing", "no image of this source was scored")
    rec = dict(stats_by_region[r])
    rec["complete"], rec["n_expected"] = complete, n_expected
    if rec["status"] == "ok" and not complete:
        rec["status"] = "partial"
        rec["reason"] = (f"{rec['n_images_scored']} of {n_expected} images scored: sources are "
                         "not comparable on different image sets, and the CI is of this subset")
    if r == "R2" and rg.R2_NEAR_EMPTY is None:
        rec["caveat"] = S6_CAVEAT
    return {**label, "dataset": ds, "split": SPLIT_OF[ds], "region": r,
            "kernel_region": kernel_name(r, ds), **rec}


def _header(root, prov_list, absent, provider, boot, ack, gt_root) -> dict:
    esha, _ = rg.eval_sha(rg.CODE_ROOT)
    gt = {}
    for ds in DATASETS:
        try:
            b = ek.gt_bundle(ds, gt_root, root)
            gt[ds] = {"bundle": b["name"], "gt_pack_sha": b["gt_pack_sha"],
                      "integrity": b["integrity"]}
        except (OSError, ValueError, rg.IntegrityError) as e:
            gt[ds] = {"bundle": rg.GT_BUNDLE[ds], "gt_pack_sha": None, "integrity": f"unknown: {e}"}
    return {
        "tool": "tools/p1/region_table.py", "plan_step": "W3-02", "cycle": "CYC-003",
        "metric_version": rg.METRIC_VERSION, "regions_provider": provider.name,
        "regions_version": provider.version,
        "fl_label": {"kitti15": "Fl", "kitti12": "Fl(K15-rule)"},
        "protocol_key": {
            "split": {"kitti15": "dev", "kitti12": "k12"},
            "protocol": "n/a: cached diagnostic predictions, no training protocol (per-source "
                        "values from the spec where given)",
            "eval_sha": esha, "gt_pack_sha": {d: gt[d]["gt_pack_sha"] for d in DATASETS},
            "agg": "per statistic: px-pooled, except the one EPE named img-mean "
                   "(images with covered pixels in the region)",
            "flags": "per source (spec)", "oracle": "per source: true for teacher rows",
            "gpu_name": "per source", "lock_hash": "per source", "packing": "per source",
            "note": "per-source key fields and pred_sha are under 'sources'"},
        "gt": gt,
        "statistics": {"unit": "image", "method": "percentile bootstrap, replicate = pooled "
                       "statistic recomputed from per-image counts (tools/eval/stats.py)",
                       **{k: boot[k] for k in ("B", "seed", "alpha", "seed_source",
                                               "settings_source")}},
        "coverage_rule": "fail-closed Fl counts an uncovered pixel as an outlier [E, DRAFT "
                         "convention of prereg/drafts/GATE-1.json]; Fl covered, EPE and coverage "
                         "are reported beside it",
        "s6": {"R2_NEAR_EMPTY": rg.R2_NEAR_EMPTY, "acknowledged": bool(ack),
               "reason": ack if isinstance(ack, str) else None,
               "caveat": S6_CAVEAT if ack and rg.R2_NEAR_EMPTY is None else None},
        "incomplete_sources": [f"{p['source']}/{p['pose']}/{p['window']}" for p in prov_list
                               if not p.get("complete")],
        "regions": list(REGIONS) + (["R4u"] if not provider.staged_reason else []),
        "sources": prov_list, "sources_absent_from_spec": absent,
        "ledger": "this tool writes no results row (hard rule 7); see the module header",
    }


# ------------------------------------------------------------------------------ rendering
def _f(x, nd=2) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def _m(rec: dict | None, key: str, nd=2) -> str:
    if not rec or rec.get(key) is None:
        return "-"
    e = rec[key]
    if e.get("value") is None:
        return "undefined"
    ci = e.get("ci")
    return f"{e['value']:.{nd}f}" + (f" [{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]" if ci else "")


def render_markdown(doc: dict) -> str:
    h = doc["header"]
    out = [MD_MARK + " (W3-02): cached predictions, dev half and KITTI-12", "",
           f"Regions: {h['regions_provider']} ({h['regions_version']}). "
           f"Metric {h['metric_version']}, "
           f"eval_sha `{h['protocol_key']['eval_sha'][:12]}`. Statistics: image bootstrap, "
           f"B={h['statistics']['B']}, seed={h['statistics']['seed']} "
           f"({h['statistics']['seed_source']}), level {1 - h['statistics']['alpha']:.2f}.", "",
           f"Coverage rule: {h['coverage_rule']}.", "",
           "Fl is `Fl(K15-rule)` on KITTI-12. EPE is in px; both EPE aggregations are named. "
           "Teacher rows are oracle rows, never student results.", ""]
    miss = [s for s in h["sources"] if s.get("status") == "missing"]
    if miss:
        out += ["Missing sources (cells listed as `missing`): "
                + ", ".join(f"{s['source']}/{s['pose']}/{s['window']}" for s in miss) + ".", ""]
    if h.get("incomplete_sources"):
        out += ["**Incomplete sources** (fewer images than the split holds; rows are `partial` and "
                "are not comparable across sources): " + ", ".join(h["incomplete_sources"]) + ".",
                ""]
    if any(c.get("caveat") for c in doc["cells"]):
        out += [f"**{S6_CAVEAT}.** Reason given: {h['s6'].get('reason')}", ""]
    if h["sources_absent_from_spec"]:
        out += ["Not in the spec: " + ", ".join(h["sources_absent_from_spec"]) + ".", ""]
    for ds in DATASETS:
        rows = [c for c in doc["cells"] if c["dataset"] == ds]
        if not rows:
            continue
        for r in h["regions"]:
            sel = [c for c in rows if c["region"] == r]
            if not sel:
                continue
            kern = kernel_name(r, ds)
            out += [f"## {ds} ({SPLIT_OF[ds]}) region {r}" + (f" (kernel {kern})" if kern and kern
                                                             != r else ""), "",
                    "| source | pose | window | status | n_img | n_px | cov % | Fl fail-closed % "
                    "[CI] | Fl covered % [CI] | EPE cov pooled [CI] | EPE cov img-mean [CI] |",
                    "|---|---|---|---|---|---|---|---|---|---|---|"]
            for c in sel:
                if c["status"] not in ("ok", "partial"):
                    note = c["status"]
                    out.append(f"| {c['source']} | {c['pose']} | {c['window']} | {note} | "
                               f"{c['n_images'] if c['n_images'] is not None else '-'} | "
                               f"{c['n_pixels'] if c['n_pixels'] is not None else '-'} "
                               "| - | - | - | - | - |")
                    continue
                tag = c["status"] if c["status"] == "ok" else (
                    f"PARTIAL {c['n_images_scored']}/{c['n_expected']}")
                if c.get("caveat"):
                    tag += " (S6 caveat: exploratory)"
                out.append(f"| {c['source']} | {c['pose']} | {c['window']} | {tag} "
                           f"| {c['n_images']} | {c['n_pixels']} | {_m(c, 'coverage', 1)} "
                           f"| {_m(c, 'fl_failclosed')} "
                           f"| {_m(c, 'fl_covered')} | {_m(c, 'epe_covered_pooled', 3)} "
                           f"| {_m(c, 'epe_covered_img_mean', 3)} |")
            out.append("")
    reasons = sorted({(c["status"], c["reason"]) for c in doc["cells"]
                      if c["status"] not in ("ok", "n=0") and c.get("reason")})
    if reasons:
        out += ["## Status notes", ""] + [f"- `{s}`: {why}" for s, why in reasons] + [""]
    return "\n".join(out)


# ------------------------------------------------------------------------------ CLI
def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if o != o else float(o)
    return o


def out_path(arg: str, root: Path, suffix: str, is_ours=None) -> Path:
    """An output file: only a NEW file (or one this tool wrote earlier) under
    <repo>/reports/p1_region/, .json or .md as asked. An allowlist compared case-insensitively on
    the resolved path (the Mac volume is case-insensitive: reports/Results.MD IS the generated
    view), so the hand-written trackers, the generated views, ledgers and reports/experiments/
    are unreachable by construction. A symlink leaf, or a symlinked directory, resolves outside
    the allowed root and is refused. is_ours(path, suffix) -> bool replaces this tool's own
    authorship test for another tool that shares the allowlist (composite_bound.py)."""
    raw = Path(arg)
    raw = raw if raw.is_absolute() else Path.cwd() / raw
    if raw.is_symlink():
        raise Fail(EXIT_USAGE, f"{arg}: a symlink is never an output path")
    p = raw.resolve()
    allowed = (root / OUT_DIR).resolve()
    if not str(p).casefold().startswith(str(allowed).casefold() + "/"):
        raise Fail(EXIT_USAGE, f"{arg}: outputs go under {allowed}/ only (an allowlist)")
    if p.suffix.casefold() != suffix:
        raise Fail(EXIT_USAGE, f"{arg}: expected a {suffix} file")
    if p.exists() and not (is_ours or _is_ours)(p, suffix):
        raise Fail(EXIT_USAGE, f"{arg}: exists and was not written by this tool: never "
                   "overwritten")
    return p


def _is_ours(p: Path, suffix: str) -> bool:
    try:
        text = p.read_text("utf-8")
        if suffix == ".md":
            return text.startswith(MD_MARK)
        return json.loads(text).get("header", {}).get("tool") == "tools/p1/region_table.py"
    except (OSError, ValueError, AttributeError):
        return False


def bootstrap_settings(a, root: Path) -> dict:
    ci = {}
    try:
        ci = json.loads((root / "prereg" / "PRIMARY.json").read_text("utf-8"))[
            "primary_endpoint"]["ci"]
    except (OSError, KeyError, ValueError):
        pass
    src = "prereg/PRIMARY.json primary_endpoint.ci" if ci else "command line"
    B = a.B if a.B is not None else ci.get("B")
    level = ci.get("level")
    alpha = a.alpha if a.alpha is not None else (None if level is None else round(1 - level, 10))
    seed = a.seed if a.seed is not None else ci.get("bootstrap_seed")
    missing = [n for n, v in (("--B", B), ("--alpha", alpha), ("--seed", seed)) if v is None]
    if missing:
        raise Fail(EXIT_USAGE, f"{' '.join(missing)} needed: prereg/PRIMARY.json does not give it "
                   "(bootstrap_seed is null until prereg-v3; the CLI value is a diagnostic one)")
    over = [n for n, v in (("B", a.B), ("alpha", a.alpha), ("seed", a.seed)) if v is not None]
    return {"B": int(B), "alpha": float(alpha), "seed": int(seed),
            "seed_source": "command line (diagnostic, not the prereg seed)" if a.seed is not None
            else "prereg/PRIMARY.json",
            "settings_source": src + (f"; overridden: {', '.join(over)}" if over else "")}


def run(a) -> int:
    root = ek.repo_root()
    gt_root = Path(a.gt_root)
    C.refuse_sealed(gt_root, a.spec)               # before the spec is even read
    spec = json.loads(Path(a.spec).read_text("utf-8"))
    boot = bootstrap_settings(a, root)
    if a.acknowledge_s6 and not (a.s6_reason or "").strip():
        raise Fail(EXIT_USAGE, "--acknowledge-s6 needs --s6-reason TEXT (recorded in the header)")
    json_out = out_path(a.json_out, root, ".json") if a.json_out else None
    md_out = out_path(a.md_out, root, ".md") if a.md_out else None
    if not gt_root.is_dir():
        raise Fail(EXIT_PRE, f"{gt_root}: GT root not found")
    provider = make_provider(a.regions, gt_root)

    def gt_open(ds: str):
        try:
            return rg.GTSource(ds, root=gt_root)
        except (OSError, ValueError) as e:
            raise Fail(EXIT_PRE, str(e)) from None

    try:
        doc = build_table(spec, root=root, gt_open=gt_open, provider=provider, boot=boot,
                          acknowledge_s6=(a.s6_reason.strip() if a.acknowledge_s6 else False),
                          gt_cache_images=a.gt_cache_images, gt_root=gt_root)
    except rg.IntegrityError as e:
        raise Fail(EXIT_INTEGRITY, str(e)) from None
    finally:
        provider.close()
    doc = _jsonable(doc)
    missing = [s for s in doc["header"]["sources"] if s.get("status") == "missing"]
    incomplete = doc["header"]["incomplete_sources"]
    text_json = json.dumps(doc, indent=1, sort_keys=False) + "\n"
    md = render_markdown(doc)
    if json_out:
        ek.atomic_write(json_out, text_json)
    if md_out:
        ek.atomic_write(md_out, md + "\n")
    if not json_out and not md_out:
        sys.stdout.write(md + "\n" if a.stdout_format == "md" else text_json)
    else:
        print(f"REGION_TABLE cells={len(doc['cells'])} missing_sources={len(missing)} "
              f"incomplete_sources={len(incomplete)} json={json_out or '-'} md={md_out or '-'}",
              file=sys.stderr)
    if incomplete and a.strict:
        raise Fail(EXIT_PRE, f"{len(incomplete)} source(s) missing, empty or incomplete under "
                   f"--strict: {', '.join(incomplete)}")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", required=True)
    ap.add_argument("--gt-root", default="/Volumes/Shin/U2FLOW_data/gt")
    ap.add_argument("--regions", default="v1", help="v1 | staged-r4 | module:PATH.py")
    ap.add_argument("--json-out")
    ap.add_argument("--md-out")
    ap.add_argument("--stdout-format", choices=("json", "md"), default="json")
    ap.add_argument("--B", type=int)
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--acknowledge-s6", action="store_true",
                    help="show R2 counts although regions.R2_NEAR_EMPTY is not fixed (S6); every "
                    "R2 cell then carries a caveat; needs --s6-reason")
    ap.add_argument("--s6-reason", help="why R2 is looked at before the S6 threshold exists")
    ap.add_argument("--strict", action="store_true",
                    help="exit 2 when a source is missing, empty or incomplete")
    ap.add_argument("--gt-cache-images", type=int, default=300)
    try:
        a = ap.parse_args(argv)
    except SystemExit as ex:
        return EXIT_USAGE if ex.code else EXIT_OK
    try:
        return run(a)
    except Fail as e:
        print(f"region_table: {'REFUSED (seal)' if e.code == EXIT_SEAL else 'error'}: {e}",
              file=sys.stderr)
        return e.code
    except C.Stop as e:
        print(f"region_table: {e}", file=sys.stderr)
        return e.code
    except rg.IntegrityError as e:
        print(f"region_table: integrity: {e}", file=sys.stderr)
        return EXIT_INTEGRITY
    except (OSError, ValueError, KeyError) as e:
        print(f"region_table: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_ERR


if __name__ == "__main__":
    sys.exit(main())
