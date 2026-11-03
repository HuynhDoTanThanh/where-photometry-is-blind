"""Shared code of tools/p1 (a package named p1lib so it never shadows tools/lib or another
package's modules in one pytest process): common.py (exit codes, atomic writes, the seal
refusal, shards, bus progress, settings stamps, weights pins), kitti.py (frames, windows,
calibration), cache.py (deterministic npz items, tar shards, manifests), jobrun.py (the
resumable item-job driver), multiproc.py (K workers), geomio.py (geometry, evidence and SAM
readers; the teacher_cache bridge), u2flow.py (the authors' loader, network and FF procedure)
and synth.py (the synthetic tree and the stub scene of the CPU tests)."""
