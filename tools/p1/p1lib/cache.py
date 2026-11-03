"""The P1 cache format: one deterministic .npz per item, packed into tar shards with a manifest.

How to use (library only):
    blob = npz_bytes({"inv_depth": a16, "pose": T}, meta={...})    # bytes, byte-deterministic
    write_item(work_dir, item_id, arrays, meta)                     # atomic <work>/<id>.npz
    arrays, meta = read_item(path_or_bytes)
    pack_shard(work_dir, ids, out_dir / "shard_007.tar")            # deterministic tar
    for item_id, arrays, meta in iter_shard(tar_path): ...
    idx = CacheIndex(cache_dir)  ->  idx.get(item_id)               # reads <kind>_manifest.json
    pack_bits(mask) / unpack_bits(packed, shape)                    # bool masks, 1 bit/pixel

Layout of a cache dir (one source, one recipe):
    <cache>/shard_###.tar    members '<id>.npz' in id order, mtime 0, uid/gid 0, mode 0644
    <cache>/manifest.json    {schema, kind, source, settings, shards: [{file, sha256, bytes,
                              items}], n_items, failures, utc}
    <cache>/failures.csv     item, stage, reason
Inside an .npz: the arrays, plus 'meta_json' (uint8 bytes of a JSON object: frames, K, grid,
units, the producing tool and recipe). No pickle anywhere (allow_pickle=False on every read).

Why it is built this way.
- Deterministic bytes: a zip with fixed dates and a USTAR tar with zeroed metadata, so the same
  arrays give the same sha256 on the VM (numpy 2.1.3) and the Mac (numpy 2.5.3); numpy's own
  savez stamps the current time on some versions [E: zipfile's default date is now()], and
  tools/eval/eval_kitti.py's ShardWriter uses the same fixed-date rule for its predictions.
- Deflate, not zstd: the plan asks for 'fp16 + zstd' [P W2-08, W2-11], but zstandard is on
  neither the frozen VM image (env/lock_G0.txt, vm-requirements.txt) nor the Mac venv, and adding
  it would move pip_freeze_sha256 and lock_hash (codebase lock-change policy). zlib is in the
  standard library; the size cost is [U] until a shard is measured.
- Items are written one by one and packed only when a whole shard is done, so a killed job
  resumes at the item, and a shard never ships half full.
"""
from __future__ import annotations

import io
import json
import os
import tarfile
import zipfile
from pathlib import Path

from . import common as C

ZIP_DATE = (1980, 1, 1, 0, 0, 0)
META_KEY = "meta_json"


def npz_bytes(arrays: dict, meta: dict | None = None, compress: bool = True) -> bytes:
    import numpy as np

    buf = io.BytesIO()
    method = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    items = dict(arrays)
    if meta is not None:
        items[META_KEY] = np.frombuffer(json.dumps(meta, sort_keys=True, default=_jdefault)
                                        .encode(), dtype=np.uint8)
    with zipfile.ZipFile(buf, "w", method) as zf:
        for name in sorted(items):
            arr = np.asarray(items[name])
            if arr.dtype == object:
                raise TypeError(f"{name}: object arrays are refused (no pickle)")
            b = io.BytesIO()
            np.lib.format.write_array(b, np.ascontiguousarray(arr), allow_pickle=False)
            zi = zipfile.ZipInfo(f"{name}.npy", date_time=ZIP_DATE)
            zi.external_attr = 0o644 << 16
            zi.compress_type = method
            zf.writestr(zi, b.getvalue())
    return buf.getvalue()


def _jdefault(o):
    import numpy as np

    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON: {type(o)}")


def read_item(src) -> tuple[dict, dict]:
    import numpy as np

    data = src if isinstance(src, (bytes, bytearray)) else Path(src).read_bytes()
    arrays, meta = {}, {}
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in zf.namelist():
            arr = np.lib.format.read_array(io.BytesIO(zf.read(name)), allow_pickle=False)
            key = name[:-4] if name.endswith(".npy") else name
            if key == META_KEY:
                meta = json.loads(arr.tobytes().decode())
            else:
                arrays[key] = arr
    return arrays, meta


def item_path(work_dir: Path, item_id: str) -> Path:
    return Path(work_dir) / f"{item_id}.npz"


def write_item(work_dir: Path, item_id: str, arrays: dict, meta: dict) -> Path:
    p = item_path(work_dir, item_id)
    C.atomic_bytes(p, npz_bytes(arrays, meta))
    return p


def pack_shard(work_dir: Path, ids: list[str], out_tar: Path) -> dict:
    out_tar.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_tar.with_name(f".{out_tar.name}.tmp{os.getpid()}")
    with tarfile.open(tmp, "w", format=tarfile.USTAR_FORMAT) as tf:
        for i in ids:
            data = item_path(work_dir, i).read_bytes()
            ti = tarfile.TarInfo(f"{i}.npz")
            ti.size, ti.mtime, ti.mode, ti.uid, ti.gid = len(data), 0, 0o644, 0, 0
            ti.uname = ti.gname = ""
            tf.addfile(ti, io.BytesIO(data))
    os.replace(tmp, out_tar)
    return {"file": out_tar.name, "sha256": C.sha256_file(out_tar),
            "bytes": out_tar.stat().st_size, "items": list(ids)}


def iter_shard(tar_path: Path):
    with tarfile.open(tar_path, "r") as tf:
        for m in tf.getmembers():
            if not m.isfile() or not m.name.endswith(".npz") or "/" in m.name:
                continue
            f = tf.extractfile(m)
            arrays, meta = read_item(f.read())
            yield m.name[:-4], arrays, meta


def shard_name(i: int) -> str:
    return f"shard_{i:03d}.tar"


def write_manifest(cache_dir: Path, doc: dict) -> Path:
    p = Path(cache_dir) / "manifest.json"
    C.atomic_json(p, doc)
    return p


class CacheIndex:
    """Random access to a cache dir by item id (verifies each shard's sha256 once)."""

    def __init__(self, cache_dir: Path, verify: bool = True):
        self.dir = Path(cache_dir)
        C.refuse_sealed(self.dir)
        man = self.dir / "manifest.json"
        if not man.is_file():
            raise C.Stop(C.EXIT_PRE, f"{self.dir}: no manifest.json")
        self.manifest = C.load_json(man)
        self.where: dict[str, str] = {}
        self.sha = {}
        for sh in self.manifest.get("shards", []):
            self.sha[sh["file"]] = sh["sha256"]
            for i in sh["items"]:
                self.where[i] = sh["file"]
        self.verify, self.checked = verify, set()
        self.cache: dict[str, dict] = {}

    def ids(self) -> list[str]:
        return sorted(self.where)

    def _load(self, fname: str) -> dict:
        if fname not in self.cache:
            p = self.dir / fname
            if self.verify and fname not in self.checked:
                if C.sha256_file(p) != self.sha[fname]:
                    raise C.Stop(C.EXIT_PRE, f"{p}: sha256 differs from manifest.json")
                self.checked.add(fname)
            self.cache = {fname: {i: (a, m) for i, a, m in iter_shard(p)}}
        return self.cache[fname]

    def get(self, item_id: str) -> tuple[dict, dict] | None:
        f = self.where.get(item_id)
        return None if f is None else self._load(f).get(item_id)


def pack_bits(mask):
    import numpy as np

    m = np.asarray(mask, dtype=bool)
    return np.packbits(m.reshape(-1))


def unpack_bits(packed, shape):
    import numpy as np

    n = int(np.prod(shape))
    return np.unpackbits(np.asarray(packed, dtype=np.uint8))[:n].reshape(shape).astype(bool)
