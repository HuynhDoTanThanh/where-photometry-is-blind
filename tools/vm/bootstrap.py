"""Run first on every Colab VM: prove the GPU is a G4 before anything else, then install extras.

How to invoke. From the Mac, always through tools/colab/acct.sh. `colab exec -f` cannot pass argv
(the file runs inside the session's IPython kernel, topaneu INS-022), so the mode travels in an
--env value. This file deletes that value from os.environ again, because --env writes into the
kernel's environment and the kernel keeps it for every later exec (execution.py:65-72, 238):

    tools/colab/acct.sh -- exec -s S -f tools/vm/bootstrap.py \
        --env U2_BOOTSTRAP_MODE=assert --timeout 120     # phase 1, before any code is staged
    tools/colab/acct.sh -- exec -s S -f tools/vm/bootstrap.py \
        --env U2_BOOTSTRAP_MODE=install --timeout 60     # phase 2, detached, after staging
    tools/colab/acct.sh -- exec -s S -f tools/vm/bootstrap.py \
        --env U2_BOOTSTRAP_MODE=status --timeout 60      # poll every 30 s until final

On the VM itself, or under `colab run` (which does pass argv):

    python tools/vm/bootstrap.py --assert | --install | --status | --emit-lock

What each mode does.
- --assert (phase 1, seconds): the GPU name contains 'RTX PRO 6000', the compute capability is
  (12, 0), total memory is at least 90 GiB, and torch lists sm_120. Prints one BOOTSTRAP_JSON line
  and then BOOTSTRAP_ASSERT_OK, SUBSTITUTED:<gpu> or ASSERT_FAIL:<reason>. Writes
  /content/u2flow_env.json (nvidia-smi read only, nproc, cgroup limits, disk, versions, the sha256
  of pip freeze) and /content/logs/nvidia-smi-q.txt.
- --install (phase 2): starts --install-foreground in a new session (setsid), logging to
  /content/logs/bootstrap.log, and returns at once with BOOTSTRAP_STARTED (or BOOTSTRAP_RUNNING).
  The foreground job (a) repairs image drift from env/torch_wheels_G0.txt when torch, torchvision,
  CUDA or cuDNN differ from env/lock_G0.txt, (b) installs env/vm-requirements.txt with --no-deps
  --require-hashes and fails if that moved torch or torchvision, (c) compares the result with the
  lock. The verdict goes to /content/u2flow/bootstrap.status: BOOTSTRAP_OK, LOCK_MISMATCH or
  PIP_FAIL.
- --status prints that verdict, or BOOTSTRAP_NONE / BOOTSTRAP_RUNNING / BOOTSTRAP_DIED.
- --emit-lock prints this VM's lock between LOCK_G0_BEGIN and LOCK_G0_END, in the format that
  env/lock_G0.txt defines (sorted key=value lines), then lock_hash=, then the pip freeze it hashed
  between LOCK_G0_FREEZE_BEGIN and LOCK_G0_FREEZE_END, and writes
  /content/u2flow_lock_candidate.txt. Collected from the G0 VM at GATE-0, the block becomes
  env/lock_G0.txt verbatim (tools/ops/gate0_evidence.py lock-write, then `lock` for the GATE-0
  lock-g0 artefact once the file is committed).

Exit codes (script and `colab run` modes; under exec only the sentinel lines cross):
    0 ok   1 error in this file   2 phase 2 still running (or started)   3 SUBSTITUTED: not a G4
    4 ASSERT_FAIL: a G4 that torch cannot drive   5 LOCK_MISMATCH, PIP_FAIL or DIED   64 bad mode

WHY IT IS BUILT THIS WAY.
- Three independent G4 checks, because each earlier one can lie. An unknown --gpu value becomes an
  A100 inside the CLI (session.py:131-139); `colab new` stores the accelerator it asked for, not
  the one it got (session.py:243-253); Pro+ users have reported silent L4/T4 substitution [U, month
  plan]. The listing's '| Hardware: G4' is the second check; this file is the third, and the only
  one that reads the silicon. Under 90 GiB on an 'RTX PRO 6000' means a MIG slice or another SKU,
  so it is a substitution too (topaneu measured 97,887 MiB, INS-032).
- SUBSTITUTED and ASSERT_FAIL are different verdicts on purpose. The first means "wrong card: stop
  the VM and back off"; the second means "right card, broken image: repair, do not back off".
- torch is probed in a fresh interpreter. A CUDA context created inside the kernel lives as long as
  the kernel does and holds VRAM that packed jobs need [E: about 0.5-1 GB per context].
- Phase 2 is detached because nothing long may run inside `colab exec`. In jupyter-kernel-client
  (read in 0.8.0; 0.9.0 is pinned and selftest checks the same code) the --timeout deadline is not enforced until the kernel goes idle
  (wsclient.py:1000-1059), so a blocking pip install holds the channel for as long as it runs;
  topaneu's blocking exec wedged the channel for 30+ min on 17 Sep until restart-kernel.
- It never runs `pip install -r requirements.txt`: the upstream file pins opencv<4, which has no
  wheel for Colab's Python 3.13 (month plan). --no-deps with --require-hashes cannot pull a new
  torch in behind our back, and the before/after comparison proves that it did not.
- The lock contract is env/lock_G0.txt's own header: every key in it counts, lock_hash is the
  sha256 of its non-comment lines, and a file still marked status=placeholder (or a
  torch_wheels_G0.txt still marked U2F-PLACEHOLDER) must be refused, never hashed or installed.
  So any difference is LOCK_MISMATCH. The status also says whether only non-core keys differ
  (driver, pip freeze, vm-requirements sha), because a Colab image update changes the freeze
  even when torch is repaired; what then happens is the codebase skill's lock-change policy.
- It never stats /content/drive: touching that path starts Colab's auto-mount flow (topaneu
  INS-036/041). A mount is detected from /proc/mounts only.
- Every write is tmp + os.replace, so a status file is either the old verdict or the new one.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

G4_NAME_TOKEN = "RTX PRO 6000"
G4_CC = (12, 0)
G4_ARCH = "sm_120"
G4_MIN_GIB = 90.0

MODE_ENV = "U2_BOOTSTRAP_MODE"
MODES = ("assert", "install", "install-foreground", "status", "emit-lock")
FLAGS = {"--" + m: m for m in MODES}

# The keys of env/lock_G0.txt, as its header defines them (sorted on write).
LOCK_KEYS = ("cc", "cuda", "cudnn", "cv2", "driver", "gpu_name", "numpy", "pip_freeze_sha256",
             "python", "sm_120_in_arch_list", "torch", "torchvision", "vm_requirements_sha256")
# The numerical stack; if only other keys differ the status says so (core_keys_match).
CORE_KEYS = ("python", "torch", "torchvision", "cuda", "cudnn", "numpy", "cv2")
# A difference here is image drift that env/torch_wheels_G0.txt can repair.
TORCH_KEYS = ("torch", "torchvision", "cuda", "cudnn")
LOCK_PLACEHOLDER = ("status", "placeholder")
WHEELS_PLACEHOLDER = "U2F-PLACEHOLDER"

SMI_FIELDS = ("name", "memory.total", "driver_version", "compute_cap", "mig.mode.current",
              "power.limit")

# Runs in a fresh interpreter so that no CUDA context is ever created inside the kernel.
_PROBE = r"""
import json, sys
out = {"python": sys.version.split()[0]}
try:
    import torch
    out["torch"] = torch.__version__
    out["cuda"] = torch.version.cuda
    try:
        out["cudnn"] = torch.backends.cudnn.version()
    except Exception:
        out["cudnn"] = None
    out["cuda_available"] = bool(torch.cuda.is_available())
    if out["cuda_available"]:
        props = torch.cuda.get_device_properties(0)
        out["name"] = torch.cuda.get_device_name(0)
        out["cc"] = list(torch.cuda.get_device_capability(0))
        out["total_bytes"] = int(props.total_memory)
        out["arch_list"] = list(torch.cuda.get_arch_list())
        out["device_count"] = int(torch.cuda.device_count())
except Exception as e:
    out["torch_error"] = type(e).__name__ + ": " + str(e)[:300]
for mod in ("torchvision", "numpy", "cv2"):
    try:
        out[mod] = __import__(mod).__version__
    except Exception:
        out[mod] = None
print("U2F_PROBE " + json.dumps(out))
"""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()  # noqa: UP017 (3.9)


def vm_root() -> Path:
    # U2_VM_ROOT exists only so the Mac can run this file against a scratch directory in tests.
    return Path(os.environ.get("U2_VM_ROOT", "/content"))


def paths() -> dict[str, Path]:
    root = vm_root()
    code = root / "u2flow"
    return {
        "root": root,
        "code": code,
        "env_json": root / "u2flow_env.json",
        "status": code / "bootstrap.status",
        "log": root / "logs" / "bootstrap.log",
        "smi_q": root / "logs" / "nvidia-smi-q.txt",
        "lock_candidate": root / "u2flow_lock_candidate.txt",
    }


def _run(cmd: list[str], timeout: float = 120.0) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        return 127, f"{type(e).__name__}: {e}"
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


# ---------------------------------------------------------------------------------------------
# Facts


def smi_facts() -> dict:
    """nvidia-smi, read only: MIG mode and the power limit are read, never changed (month plan)."""
    if shutil.which("nvidia-smi") is None:
        return {"present": False}
    fields = list(SMI_FIELDS)
    out = ""
    # Older drivers reject an unknown query field for the whole call, so drop optional ones in turn.
    for drop in (None, "compute_cap", "mig.mode.current", "power.limit"):
        if drop is not None:
            fields.remove(drop)
        rc, out = _run(["nvidia-smi", "--query-gpu=" + ",".join(fields),
                        "--format=csv,noheader,nounits"], timeout=30)
        lines = [ln for ln in out.strip().splitlines() if ln.strip()]
        if rc == 0 and lines:
            facts: dict = {"present": True, "count": len(lines)}
            facts.update(zip(fields, (v.strip() for v in lines[0].split(",")), strict=False))
            return facts
    return {"present": True, "error": out.strip()[:300]}


def probe() -> dict:
    # 300 s: the first torch import on a fresh VM can take tens of seconds.
    rc, out = _run([sys.executable, "-c", _PROBE], timeout=300)
    for line in out.splitlines():
        if line.startswith("U2F_PROBE "):
            try:
                return json.loads(line[len("U2F_PROBE "):])
            except ValueError:
                break
    return {"probe_error": f"rc={rc}: {out.strip()[-300:]}"}


def pip_freeze_lines() -> list[str] | None:
    """`pip freeze --all`, sorted, stripped, WARNING lines dropped; None when pip failed."""
    rc, out = _run([sys.executable, "-m", "pip", "freeze", "--all",
                    "--disable-pip-version-check"], timeout=180)
    if rc != 0:
        return None
    return sorted(ln.strip() for ln in out.splitlines()
                  if ln.strip() and not ln.startswith("WARNING"))


def freeze_sha_of_lines(lines: list[str] | None) -> str:
    return "" if lines is None else hashlib.sha256("\n".join(lines).encode()).hexdigest()


def pip_freeze_sha() -> str:
    return freeze_sha_of_lines(pip_freeze_lines())


def cgroup_limits() -> dict:
    """cgroup v2 first, v1 second. nproc can report CPUs the container may not use (month plan)."""
    out: dict = {}
    cpu_max = _read("/sys/fs/cgroup/cpu.max")
    if cpu_max:
        out["cpu_max"] = cpu_max
        quota, _, period = cpu_max.partition(" ")
        if quota.isdigit() and period.isdigit() and int(period) > 0:
            out["cpu_quota"] = round(int(quota) / int(period), 2)
    else:
        q = _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        p = _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if q and p and q.isdigit() and p.isdigit() and int(p) > 0:
            out["cpu_quota"] = round(int(q) / int(p), 2)
    mem = _read("/sys/fs/cgroup/memory.max") or _read("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if mem:
        out["memory_max"] = mem
        if mem.isdigit() and int(mem) < (1 << 60):
            out["memory_max_gib"] = round(int(mem) / 2**30, 1)
    return out


def meminfo_gib() -> dict:
    out = {}
    for line in (_read("/proc/meminfo") or "").splitlines():
        key, _, rest = line.partition(":")
        if key in ("MemTotal", "MemAvailable") and rest.split():
            out[key] = round(int(rest.split()[0]) / 2**20, 1)  # kB -> GiB
    return out


def drive_mounted() -> bool:
    # /proc/mounts only; stat-ing /content/drive would start Colab's auto-mount flow.
    for line in (_read("/proc/mounts") or "").splitlines():
        fields = line.split()
        if len(fields) > 1 and fields[1].startswith("/content/drive"):
            return True
    return False


def host_facts() -> dict:
    facts: dict = {"nproc": os.cpu_count(), "cgroup": cgroup_limits(), "mem_gib": meminfo_gib(),
                   "drive_mounted": drive_mounted()}
    if hasattr(os, "sched_getaffinity"):
        facts["affinity_cpus"] = len(os.sched_getaffinity(0))
    for label, p in (("content", vm_root()), ("shm", Path("/dev/shm"))):
        if p.exists():
            du = shutil.disk_usage(str(p))
            facts[f"disk_{label}_free_gb"] = round(du.free / 1e9, 1)
            facts[f"disk_{label}_total_gb"] = round(du.total / 1e9, 1)
    return facts


def _cc_from_smi(value: object) -> tuple[int, int] | None:
    try:
        major, minor = str(value).strip().split(".")
        return int(major), int(minor)
    except (ValueError, AttributeError):
        return None


def _total_gib(smi: dict, p: dict) -> float | None:
    if p.get("total_bytes"):
        return int(p["total_bytes"]) / 2**30
    try:
        return float(smi["memory.total"]) / 1024.0  # MiB with nounits
    except (KeyError, ValueError, TypeError):
        return None


def g4_verdict(smi: dict, p: dict) -> tuple[str, int]:
    """(sentinel, exit code). Hardware that is not a G4 is a substitution; a G4 that torch cannot
    drive is ASSERT_FAIL, which the Mac must treat as image drift, not as a reason to back off."""
    name = " ".join(str(p.get("name") or smi.get("name") or "none").split()) or "none"
    if not smi.get("present") and not p.get("cuda_available"):
        return "SUBSTITUTED:none", 3
    if G4_NAME_TOKEN.lower() not in name.lower():
        return f"SUBSTITUTED:{name}", 3
    cc = tuple(p["cc"]) if p.get("cc") else _cc_from_smi(smi.get("compute_cap"))
    if cc is not None and tuple(cc) != G4_CC:
        return f"SUBSTITUTED:{name} cc={cc[0]}.{cc[1]}", 3
    gib = _total_gib(smi, p)
    if gib is not None and gib < G4_MIN_GIB:
        return f"SUBSTITUTED:{name} mem_gib={gib:.1f}", 3
    if not p.get("cuda_available"):
        return "ASSERT_FAIL:torch-cannot-use-cuda", 4
    if cc is None or gib is None:
        return "ASSERT_FAIL:capability-or-memory-unknown", 4
    if G4_ARCH not in (p.get("arch_list") or []):
        return f"ASSERT_FAIL:torch-lacks-{G4_ARCH}", 4
    return "BOOTSTRAP_ASSERT_OK", 0


# ---------------------------------------------------------------------------------------------
# The lock (env/lock_G0.txt): the format is defined by that file's header, not here.


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def lock_values(p: dict, smi: dict, req: Path | None = None,
                freeze: list[str] | None = None) -> dict[str, str]:
    """The lock's keys on this VM. `freeze` (pip_freeze_lines()) is hashed when given, so the
    lock and the freeze block --emit-lock prints come from one pip call."""
    cc = p.get("cc") or _cc_from_smi(smi.get("compute_cap"))
    values = {k: ("" if p.get(k) is None else str(p.get(k))) for k in CORE_KEYS}
    values.update({
        "cc": f"{cc[0]}.{cc[1]}" if cc else "",
        "driver": str(smi.get("driver_version", "")),
        "gpu_name": " ".join(str(p.get("name") or smi.get("name") or "").split()),
        "pip_freeze_sha256": (freeze_sha_of_lines(freeze) if freeze is not None
                              else pip_freeze_sha()),
        "sm_120_in_arch_list": "true" if G4_ARCH in (p.get("arch_list") or []) else "false",
        "vm_requirements_sha256": _sha256_file(req) if req is not None and req.is_file() else "",
    })
    return values


def lock_lines(values: dict) -> list[str]:
    return [f"{k}={values.get(k, '')}" for k in sorted(LOCK_KEYS)]


def lock_hash_of_lines(lines: list[str]) -> str:
    """sha256 of the non-comment, non-empty lines, stripped and joined by '\n' (no trailing
    newline). Hashing env/lock_G0.txt and hashing a VM's emitted block give the same value."""
    body = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    return hashlib.sha256("\n".join(body).encode()).hexdigest()


def format_lock(values: dict) -> str:
    head = [f"# emitted by tools/vm/bootstrap.py --emit-lock at {utc_now()} on "
            f"{values.get('gpu_name') or 'an unknown GPU'}; copy verbatim into env/lock_G0.txt"]
    return "\n".join(head + lock_lines(values)) + "\n"


def read_lock(path: Path) -> tuple[dict[str, str], bool]:
    """(values, is_placeholder). A placeholder file is never hashed or compared (its header)."""
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out, out.get(LOCK_PLACEHOLDER[0]) == LOCK_PLACEHOLDER[1]


def mismatches(lock: dict, values: dict, keys: tuple[str, ...] = LOCK_KEYS) -> dict:
    return {k: {"lock": lock.get(k, ""), "vm": values.get(k, "")}
            for k in keys if lock.get(k, "") != values.get(k, "")}


# ---------------------------------------------------------------------------------------------
# Modes


def do_assert() -> int:
    P = paths()
    smi = smi_facts()
    p = probe()
    sentinel, rc = g4_verdict(smi, p)
    facts = {"utc": utc_now(), "verdict": sentinel, "gpu_smi": smi, "torch": p,
             "host": host_facts(), "pip_freeze_sha256": pip_freeze_sha()}
    try:
        if smi.get("present"):
            rcq, q = _run(["nvidia-smi", "-q"], timeout=60)  # read only
            if rcq == 0:
                _atomic_write(P["smi_q"], q)
        _atomic_write(P["env_json"], json.dumps(facts, indent=1, sort_keys=True) + "\n")
    except OSError as e:
        facts["write_error"] = str(e)[:200]
    print("BOOTSTRAP_JSON " + json.dumps(facts, sort_keys=True, separators=(",", ":")))
    if rc == 0:
        gib = _total_gib(smi, p) or 0.0
        print(f"BOOTSTRAP_ASSERT_OK gpu={p.get('name')!r} cc={G4_CC[0]}.{G4_CC[1]} "
              f"mem_gib={gib:.1f} driver={smi.get('driver_version', '?')}")
    else:
        print(sentinel)
    return rc


def _status_read() -> dict | None:
    try:
        return json.loads(paths()["status"].read_text())
    except (OSError, ValueError):
        return None


def _write_status(rec: dict) -> None:
    _atomic_write(paths()["status"], json.dumps(rec, sort_keys=True) + "\n")


def _alive(pid: object) -> bool:
    """kill -0 succeeds on a zombie, and the kernel never reaps the detached child, so read the
    process state as well."""
    try:
        pid_i = int(pid)  # type: ignore[arg-type]
        os.kill(pid_i, 0)
    except (OSError, ValueError, TypeError):
        return False
    stat = _read(f"/proc/{pid_i}/stat")
    if stat and ")" in stat:
        state = stat.rsplit(")", 1)[1].split()
        if state and state[0] == "Z":
            return False
    return True


def do_install() -> int:
    P = paths()
    me = P["code"] / "tools" / "vm" / "bootstrap.py"
    if not me.is_file():
        print(f"PIP_FAIL no staged code at {P['code']}: stage the code before phase 2")
        return 5
    st = _status_read()
    if st and st.get("status") == "RUNNING" and _alive(st.get("pid")):
        print(f"BOOTSTRAP_RUNNING pid={st.get('pid')} log={P['log']}")
        return 2
    P["log"].parent.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k != MODE_ENV}
    with open(P["log"], "a") as log:
        proc = subprocess.Popen([sys.executable, "-u", str(me), "--install-foreground"],
                                cwd=str(P["code"]), stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, env=env, start_new_session=True)
    _write_status({"status": "RUNNING", "pid": proc.pid, "utc": utc_now()})
    print(f"BOOTSTRAP_STARTED pid={proc.pid} log={P['log']}")
    return 2


def _pip(args: list[str]) -> int:
    cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-input",
           *args]
    print("+ " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    return proc.wait()


def _finish(status: str, **info: object) -> int:
    rec = {"status": status, "utc": utc_now(), "pid": os.getpid(), **info}
    _write_status(rec)
    print(status + " " + json.dumps(rec, sort_keys=True), flush=True)
    return 0 if status == "BOOTSTRAP_OK" else 5


def do_install_foreground() -> int:
    P = paths()
    env_dir = P["code"] / "env"
    req, lock_file, wheels = (env_dir / "vm-requirements.txt", env_dir / "lock_G0.txt",
                              env_dir / "torch_wheels_G0.txt")
    _write_status({"status": "RUNNING", "pid": os.getpid(), "utc": utc_now()})
    print(f"bootstrap phase 2 started {utc_now()} in {P['code']}", flush=True)
    if not req.is_file():
        return _finish("PIP_FAIL", reason=f"{req} is missing from the staged code")
    lock, placeholder = read_lock(lock_file) if lock_file.is_file() else ({}, True)
    wheels_ok = wheels.is_file() and WHEELS_PLACEHOLDER not in wheels.read_text()
    smi = smi_facts()
    v0 = lock_values(probe(), smi, req)
    repaired, notes = None, []
    drift = {} if placeholder else mismatches(lock, v0, TORCH_KEYS)
    if drift and not wheels_ok:
        notes.append(f"image drift in {sorted(drift)}; {wheels.name} is missing or still a "
                     "placeholder, so nothing was reinstalled")
    elif drift:
        rc = _pip(["--no-deps", "--require-hashes", "-r", str(wheels)])
        if rc != 0:
            return _finish("PIP_FAIL", reason=f"torch wheel repair exited {rc}", drift=drift)
        v1 = lock_values(probe(), smi, req)
        repaired = {"from": {k: v0.get(k) for k in TORCH_KEYS},
                    "to": {k: v1.get(k) for k in TORCH_KEYS}}
        v0 = v1
    before = {k: v0.get(k) for k in ("torch", "torchvision")}
    rc = _pip(["--no-deps", "--require-hashes", "-r", str(req)])
    if rc != 0:
        return _finish("PIP_FAIL", reason=f"pip exited {rc} on {req.name}",
                       image_drift_repaired=repaired)
    v2 = lock_values(probe(), smi, req)
    after = {k: v2.get(k) for k in ("torch", "torchvision")}
    if after != before:
        return _finish("PIP_FAIL", reason=f"{req.name} changed torch or torchvision",
                       before=before, after=after)
    info = {"versions": v2, "vm_lock_hash": lock_hash_of_lines(lock_lines(v2)),
            "image_drift_repaired": repaired, "notes": notes}
    if placeholder:
        return _finish("BOOTSTRAP_OK", lock="placeholder: no run is confirmatory before GATE-0",
                       **info)
    diff = mismatches(lock, v2)
    info["lock_hash"] = lock_hash_of_lines(lock_file.read_text().splitlines())
    if diff:
        # A dated lock (env/lock_G0.<yyyymmdd>.txt) is a COMMITTED acceptance of a later Colab image
        # (codebase skill section 7: parity check + same-seed smoke passed); a VM that matches one
        # of them completely is fine, and its rows carry that lock's hash. 2026-10-06: the image
        # moved (pip freeze only) and every launch died with LOCK_MISMATCH until this existed.
        for dated in sorted(env_dir.glob("lock_G0.[0-9]*.txt")):
            if "pip-freeze" in dated.name:
                continue
            dlock, dplace = read_lock(dated)
            if not dplace and not mismatches(dlock, v2):
                info["lock_hash"] = lock_hash_of_lines(dated.read_text().splitlines())
                return _finish("BOOTSTRAP_OK", lock=f"match {dated.name}", **info)
    if diff:
        return _finish("LOCK_MISMATCH", mismatches=diff,
                       core_keys_match=not any(k in diff for k in CORE_KEYS), **info)
    return _finish("BOOTSTRAP_OK", lock="match", **info)


def do_status() -> int:
    st = _status_read()
    if st is None:
        print("BOOTSTRAP_NONE phase 2 has not run on this VM since the code was last staged")
        return 1
    status = str(st.get("status"))
    if status == "RUNNING":
        if _alive(st.get("pid")):
            print(f"BOOTSTRAP_RUNNING pid={st.get('pid')} log={paths()['log']}")
            return 2
        print(f"BOOTSTRAP_DIED pid={st.get('pid')} ended without a verdict; read {paths()['log']}")
        return 5
    print(status + " " + json.dumps(st, sort_keys=True))
    return 0 if status == "BOOTSTRAP_OK" else 5


def do_emit_lock() -> int:
    """The lock block, then the pip freeze it hashes (LOCK_G0_FREEZE_BEGIN/END, one requirement
    per line), so the Mac can store the freeze beside the GATE-0 lock-g0 artefact and fill
    env/torch_wheels_G0.txt from it (tools/ops/gate0_evidence.py lock-write; codebase section 7).
    Run after phase 2 on the G0 VM: vm_requirements_sha256 is read from the staged code."""
    freeze = pip_freeze_lines()
    text = format_lock(lock_values(probe(), smi_facts(), paths()["code"] / "env"
                                   / "vm-requirements.txt", freeze))
    try:
        _atomic_write(paths()["lock_candidate"], text)
    except OSError as e:
        print(f"# could not write {paths()['lock_candidate']}: {e}")
    print("LOCK_G0_BEGIN")
    print(text, end="")
    print("LOCK_G0_END")
    print("lock_hash=" + lock_hash_of_lines(text.splitlines()))
    if freeze is not None:
        print("LOCK_G0_FREEZE_BEGIN")
        for line in freeze:
            print(line)
        print("LOCK_G0_FREEZE_END")
    return 0


def resolve_mode(argv: list[str]) -> str:
    env_mode = os.environ.pop(MODE_ENV, None)  # always removed: --env persists in the kernel
    for arg in argv[1:]:
        if arg in FLAGS:
            return FLAGS[arg]
        if arg in ("-h", "--help"):
            return "help"
    if env_mode:
        return env_mode if env_mode in MODES else "bad:" + env_mode
    return "assert"


def main(argv: list[str]) -> int:
    mode = resolve_mode(argv)
    if mode == "help":
        print(__doc__)
        return 0
    if mode.startswith("bad:"):
        print(f"BOOTSTRAP_BAD_MODE {mode[4:]!r}: expected one of {', '.join(MODES)}")
        return 64
    handlers = {"assert": do_assert, "install": do_install,
                "install-foreground": do_install_foreground, "status": do_status,
                "emit-lock": do_emit_lock}
    try:
        return handlers[mode]()
    except Exception as e:  # a traceback under exec would hide the sentinel the Mac waits for
        print(f"BOOTSTRAP_ERROR {type(e).__name__}: {e}")
        return 1


def _under_exec() -> bool:
    # `colab exec -f` runs this text in the session's IPython kernel with the kernel's argv, where
    # SystemExit prints a traceback (INS-022); `colab run` sets argv[0] to this file's basename.
    return "ipykernel" in sys.modules and not os.path.basename(sys.argv[0]).startswith("bootstrap")


if __name__ == "__main__":
    _rc = main(sys.argv)
    if not _under_exec():
        raise SystemExit(_rc)
