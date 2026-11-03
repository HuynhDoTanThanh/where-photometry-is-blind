"""The item-job driver every P1 producer shares: shard selection, skip-if-done, per-item
failures, shard packing, the manifest, progress on the bus and a clean stop on TERM.

How to use (library only):
    job = ItemJob(kind="geometry", source="mapanything", items=rows, cache_dir=out/"cache"/...,
                  work_dir=W, shard_size=25, shard="0/1", identity={...}, settings={...},
                  allow_failures=False)
    job.progress.resolved({...})      # after the inputs are checked (launch.sh waits for it)
    rc = job.run(produce)             # produce(row) -> (arrays, meta); raises ItemFailed
Output lines (the P1 tools say 'as p1lib/jobrun.py'): P1_RESOLVED or RESOLVED_CONFIG (via the
bus), P1_PROGRESS, P1_ITEM_FAILED, P1_SHARD, P1_DONE, P1_INCOMPLETE, P1_STOPPED, P1_ERROR.
Exit codes returned by run(): 0 every owned shard packed and no failure · 1 items left or failed
(a rerun skips done items; --retry-failed redoes recorded failures) · 143 stopped by TERM.

Why it is built this way.
- A shard is packed only when every one of its items is done (a success or, with
  allow_failures, a recorded failure), so a shard never ships half full and a rerun resumes at
  the item (sam_masks.py's per-frame rule, applied per window).
- Shard indices are global (shard i holds items[i*S:(i+1)*S]); `--shard k/n` owns i % n == k, so
  n VMs write disjoint shard files that merge by copying (cache.merge_dirs), and a rerun with
  the same k/n owns the same shards.
- COLMAP and self-triangulation fail on some windows by nature (plan W2-12: 'Cửa sổ thất bại
  ghi vào failures.csv'): allow_failures records an ItemFailed in <id>.failed.json and the
  manifest; an unexpected exception is recorded too but still makes the run exit 1, and a
  C.Stop (a precondition or refusal) stops the whole job with its own code. A GPU teacher
  failing is a bug, so its default is to exit 1.
- The manifest is rewritten after every packed shard from what is on disk, never from memory, so
  a killed job leaves a manifest that describes exactly the shards present; a cache dir whose
  manifest names another recipe is refused (exit 3), as a work dir with another stamp is.
"""
from __future__ import annotations

import json
import signal
import time
import traceback
from pathlib import Path

from . import cache as CA
from . import common as C


class ItemFailed(Exception):
    """One item could not be produced; stage names where."""

    def __init__(self, stage: str, reason: str):
        super().__init__(f"{stage}: {reason}")
        self.stage, self.reason = stage, reason


class ItemJob:
    def __init__(self, *, kind: str, source: str, items: list[dict], cache_dir: Path,
                 work_dir: Path, shard_size: int, shard: str | None, identity: dict,
                 settings: dict, allow_failures: bool = False, retry_failed: bool = False,
                 limit: int | None = None, selection: dict | None = None):
        C.refuse_sealed(cache_dir, work_dir)
        self.kind, self.source = kind, source
        self.items = items
        self.ids = [r["id"] for r in items]
        if len(set(self.ids)) != len(self.ids):
            raise C.Stop(C.EXIT_USAGE, "duplicate item ids")
        self.cache_dir, self.work_dir = Path(cache_dir), Path(work_dir)
        self.size = int(shard_size)
        self.k, self.n = C.parse_shard(shard)
        self.identity, self.settings = identity, settings
        self.allow_failures, self.retry_failed = allow_failures, retry_failed
        # a declared subset of the full list (run_ff_shard --subset): the manifest says so, and
        # its `complete` (what GATE-1 G1.P3 reads: the FULL list) stays false
        self.selection = selection
        self.shards = C.shard_plan(self.ids, self.size)
        self.mine = C.mine(len(self.shards), self.k, self.n)
        if limit is not None:  # a bench: only the first `limit` items of the owned shards
            keep, left = [], int(limit)
            for i in self.mine:
                if left <= 0:
                    break
                keep.append(i)
                left -= len(self.shards[i])
            self.mine = keep
        self.total = sum(len(self.shards[i]) for i in self.mine)
        self.progress = C.Progress(self.total)
        self.stop = None
        self.timings: list[float] = []
        self.host = C.host_record()  # provenance in the manifest, never in the identity

    # ---------------------------------------------------------------- bookkeeping
    def failed_path(self, item_id: str) -> Path:
        return self.work_dir / f"{item_id}.failed.json"

    def done(self, item_id: str) -> bool:
        return CA.item_path(self.work_dir, item_id).is_file() or (
            self.allow_failures and self.failed_path(item_id).is_file())

    def _on_term(self, signum, frame) -> None:  # noqa: ARG002
        self.stop = signum

    def failures(self) -> list[dict]:
        out = []
        for p in sorted(self.work_dir.glob("*.failed.json")):
            try:
                out.append(json.loads(p.read_text()))
            except (OSError, ValueError):
                continue
        return out

    def write_manifest(self) -> None:
        shards = []
        for i, ids in enumerate(self.shards):
            tar = self.cache_dir / CA.shard_name(i)
            if not tar.is_file():
                continue
            meta = self.work_dir / f".{CA.shard_name(i)}.json"
            doc = C.load_json(meta) if meta.is_file() else None
            if doc is None or doc.get("sha256") != C.sha256_file(tar):
                doc = {"file": tar.name, "sha256": C.sha256_file(tar),
                       "bytes": tar.stat().st_size,
                       "items": [x for x in ids if CA.item_path(self.work_dir, x).is_file()]}
            shards.append(doc)
        fails = [f for f in self.failures() if f.get("item") in set(self.ids)]
        C.write_failures(self.cache_dir / "failures.csv", fails)
        n_items = sum(len(s["items"]) for s in shards)
        CA.write_manifest(self.cache_dir, {
            "schema": 1, "kind": self.kind, "source": self.source, "settings": self.settings,
            "identity": self.identity, "shard_size": self.size, "n_shards_planned":
            len(self.shards), "shard": f"{self.k}/{self.n}", "shards": shards,
            "n_items": n_items, "n_items_planned": len(self.ids),
            # every item of the WHOLE list present, none failed: what GATE-1 G1.P3 reads
            # (prereg/drafts/GATE-1.json); a shard k/n alone is complete only after merge-cache
            "complete": self.selection is None and n_items == len(self.ids) and not fails,
            "complete_selection": n_items == len(self.ids) and not fails,
            "selection": self.selection,
            "host": self.host,
            "failures": [{"item": f["item"], "stage": f["stage"]} for f in fails],
            "s_per_item_median": (sorted(self.timings)[len(self.timings) // 2]
                                  if self.timings else None),
            "utc": C.utc_now()})

    # ---------------------------------------------------------------- the loop
    def check_cache_dir(self) -> None:
        """Stop(3) when the cache dir already holds another recipe's manifest (two jobs aimed
        at one dir would otherwise describe each other's shards)."""
        man = self.cache_dir / "manifest.json"
        if man.is_file() and C.load_json(man).get("identity") != self.identity:
            raise C.Stop(C.EXIT_REFUSED, f"{self.cache_dir} holds another recipe's cache "
                                         "(manifest identity differs); use another --out-dir")

    def run(self, produce) -> int:
        C.check_stamp(self.work_dir, self.identity)
        self.check_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.retry_failed:
            for p in self.work_dir.glob("*.failed.json"):
                p.unlink()
        old = signal.signal(signal.SIGTERM, self._on_term)
        by_id = {r["id"]: r for r in self.items}
        finished = failed = 0
        try:
            for i in self.mine:
                ids = self.shards[i]
                tar = self.cache_dir / CA.shard_name(i)
                for item_id in ids:
                    if self.stop is not None:
                        break
                    if self.done(item_id):
                        finished += 1
                        failed += int(self.failed_path(item_id).is_file())
                        continue
                    t0 = time.time()
                    try:
                        arrays, meta = produce(by_id[item_id])
                        meta = dict(meta, item=item_id, source=self.source, kind=self.kind,
                                    seconds=round(time.time() - t0, 3))
                        CA.write_item(self.work_dir, item_id, arrays, meta)
                        self.failed_path(item_id).unlink(missing_ok=True)  # a retry succeeded
                        self.timings.append(time.time() - t0)
                    except C.Stop:
                        raise  # a precondition or refusal stops the job with its own code
                    except ItemFailed as e:
                        failed += 1
                        C.atomic_json(self.failed_path(item_id), {
                            "item": item_id, "stage": e.stage, "reason": e.reason,
                            "utc": C.utc_now()})
                        C.say("P1_ITEM_FAILED", f"{item_id} {e.stage}: {e.reason[:200]}")
                    except Exception as e:  # noqa: BLE001  recorded, then the job decides
                        failed += 1
                        C.atomic_json(self.failed_path(item_id), {
                            "item": item_id, "stage": "unexpected",
                            "reason": f"{type(e).__name__}: {e}",
                            "trace": traceback.format_exc()[-2000:], "utc": C.utc_now()})
                        C.say("P1_ITEM_FAILED", f"{item_id} unexpected: {type(e).__name__}: {e}")
                    finished += 1
                    self.progress.tick(finished, failed, shards_done=self._n_packed())
                if self.stop is not None:
                    break
                if all(self.done(x) for x in ids) and not tar.is_file():
                    ok_ids = [x for x in ids if CA.item_path(self.work_dir, x).is_file()]
                    if ok_ids and (self.allow_failures or len(ok_ids) == len(ids)):
                        doc = CA.pack_shard(self.work_dir, ok_ids, tar)
                        C.atomic_json(self.work_dir / f".{tar.name}.json", doc)
                        C.say("P1_SHARD", f"{tar.name} items={len(ok_ids)} "
                                          f"sha256={doc['sha256']}")
                    self.write_manifest()
        finally:
            signal.signal(signal.SIGTERM, old)
        self.write_manifest()
        self.progress.tick(finished, failed, force=True, shards_done=self._n_packed())
        if self.stop is not None:
            C.say("P1_STOPPED", f"signal {self.stop}; done items are kept")
            return C.EXIT_TERM
        left = [x for i in self.mine for x in self.shards[i] if not self.done(x)]
        hard_fail = [x for i in self.mine for x in self.shards[i]
                     if self.failed_path(x).is_file() and (not self.allow_failures or C.load_json(
                         self.failed_path(x)).get("stage") == "unexpected")]
        unpacked = [i for i in self.mine if not (self.cache_dir / CA.shard_name(i)).is_file()
                    and any(CA.item_path(self.work_dir, x).is_file() for x in self.shards[i])]
        C.say("P1_DONE" if not (left or hard_fail or unpacked) else "P1_INCOMPLETE",
              f"shards={len(self.mine)} items={self.total} failed={failed} left={len(left)}")
        return C.EXIT_OK if not (left or hard_fail or unpacked) else C.EXIT_FAIL

    def _n_packed(self) -> int:
        return sum(1 for i in self.mine if (self.cache_dir / CA.shard_name(i)).is_file())
