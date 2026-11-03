# Known limitations of the pre-registration (prereg-v1 draft)

What this file is. Each limitation below is known before the first pre-registration tag, says
what it can bias, and names the evidence. The paper's limitations section is drawn from it.
prereg-v1 freezes this text with the rest of prereg/ (tools/prereg/tag.sh); later findings are
added in a new version, never edited in place. Evidence tags follow AGENTS.md hard rule 6:
[M] measured here, [V file:line] read in source, [A4] the 2026-09-24 data audit, [E] reasoned,
[U] unverified.

## Exposure of the KITTI-15 training pairs before the split

**L1. Numbers on all 200 KITTI-15 training pairs existed before this pre-registration.**
- On 2026-09-24/25, before any split existed, a Mac audit of the RELEASED U²Flow checkpoints
  (fp32 on MPS) computed all-200 aggregates (KITTI-15 EPE, Fl-all, Fl-noc, Fl-occ, with and
  without +FF; KITTI-12; Sintel) and, for +FF, per-pair EPE and outlier counts on all 200
  pairs [M; reports/PLAN.md section 4, 2026-09-25].
- The per-pair rows (results/ff_kitti_per_pair.json) and the audit's extracted GT tree
  (data_scene_flow/training: flow_occ, flow_noc, obj_map, disp_*, all 200 ids) were moved to
  /Volumes/Shin/U2FLOW_audit_private (never uploaded, never used for a decision; not viewed
  per image) [prereg/inputs/split_seed.json exposure_check]. That folder lies OUTSIDE
  U2FLOW_data/sealed, so hard rule 4 does not hold while it exists [Tier 1 review,
  2026-10-02]. Since then: no tool reads it (the codebase parity test takes its frames from
  the public bundle u2f-mv-f0912-train-v1), and `seal.py plan|finalize|verify --deep` scan it
  (and the repo's output/) and exit 50 while it holds held-out GT or per-pair files. Moving the
  copy under U2FLOW_data/sealed/ or deleting it is the user's decision (AGENTS.md section 4); the
  outcome is logged in reports/PLAN.md section 4 and recorded here in the next version.
- No per-image or per-region KITTI-15 number exists for any model of ours.
- GATE-0 itself makes the released checkpoint's held-out AGGREGATE known: G0.1 records the
  all-200 EPE (image mean over 200) and Fl with n_pixels, G0.2 the dev-half rows with n_pixels,
  so held-out EPE = 2 x all200 - dev and held-out Fl follows from the counts; the same holds for
  each declared +FF seed (G0.3). This is the GATE-0 exception of hard rule 4 (released
  checkpoint only, aggregate only, once per metric and seed); no tool computes the difference.
- The GATE-0 targets and tolerances (prereg/GATE-0.json) come from the paper's published numbers
  and the plan, not from the audit; still, the audit numbers were in view when GATE-0.json was
  drafted. GATE-0 re-measures on a G4 with the frozen evaluator.
- What it can bias: the split could in principle be chosen with per-pair difficulty in view. The
  seed was fixed by the user's choice (20261002, the GATE-0 date) before the split was drawn, and
  make_split.py refuses any other seed on the real mapping [V tools/prereg/make_split.py].

**L2. One scratch seed was drawn on the real mapping.** On 2026-09-24 one scratch seed (value not
recorded) was run once on the real devkit mapping to check the algorithm's shape; nothing from
it was kept [prereg/inputs/split_seed.json excluded_note].

## The split

**L3. 58 sequences are not scene-disjoint.** The devkit's train_mapping.txt maps 142 of the 200
training sequences to 29 raw drives and leaves 58 empty [A4; V make_split.py docstring]. Whole
drives go to one half, so the 142 mapped sequences are drive-disjoint; the 58 unmapped ones
(29 per half in split v1 [M splits/k15_split_v1.json composition]) are assigned by a seeded
shuffle, and scene overlap between halves is possible for them. What it can bias: held-out
numbers may partly measure scenes seen in dev tuning. The drive-clustered bootstrap is a
robustness row only, and how it clusters the unmapped sequences is fixed in PRIMARY.json.

**L4. The held-out half has about 14-15 drives** (14 mapped drives + 29 unmapped sequences in
split v1 [M]). A drive-level interval is too coarse to carry inference; the primary CI resamples
images (metric statistics.md).

## The released U²Flow checkpoint (the baseline and the initialisation)

**L5. U²Flow's training saw KITTI test-sequence frames and the neighbouring frames of every
KITTI-15 training scene** (unlabelled multi-view frames) [reports/PLAN.md section 4,
2026-09-25: undisclosed practices; proposal datasets[0].notes]. Every arm initialised from
kitti.pth inherits that exposure, the held-out half included. HO-tr removes our own training
signal on held-out scenes, not the initialisation's.

**L6. 'best' may be chosen on the evaluated pairs.** train.py validates on all 200 KITTI-15
training pairs every val_freq steps and saves 'best' by that EPE [V 152bb38:train.py:411, 419-421]. Whether
the released kitti.pth is the final or the 'best' checkpoint is [U] (a question for the authors,
HANDOFF item 9). If 'best', every kitti.pth number on KITTI-15 train carries a selection bias.
Our runs report the final step only.

**L7. +FF is a different procedure and is unseeded.** The released +FF scripts die at import
[V train_fusion_*.py:11; AGENTS.md section 7]; the codebase patch fixes the import. Per-image
optimisation has no seed upstream, so +FF numbers are a mean over FF seeds with their spread
(GATE-0 G0.3), labelled '3-frame + per-image optimisation'. Its KITTI and Sintel
hyperparameters differ (iters 2000 / thr 35 vs 2700 / 45) and how they were chosen is [U].

**L8. Seed 1234 is hard-coded upstream** with cudnn deterministic and benchmark both on
[V 152bb38:train.py:466-472]; the codebase port adds real seeding. σ_seed includes GPU nondeterminism.

## Evaluator and metrics (kept frozen: metric skill section 1)

**L9. U²Flow's EPE is an image mean; noc/occ EPE append zeros** for images without such pixels
[V evaluate.py:559-562, 569-570], and Sintel OCC does the same [V evaluate.py:158-163]. AUSE is
per image and normalised per image; CC is a mean of per-image Spearman ρ [V evaluate.py:371-399].
We keep these semantics for comparability and add pooled Fl from counts; such numbers are not
comparable to pooled or absolute definitions elsewhere without re-computation.

**L10. KITTI-12 `fl` is the KITTI-15 outlier rule**, not the official Out-All
[V evaluate.py:860]; we label it Fl(K15-rule) and add Out-Noc / Out-All.

## Data provenance

**L11. HD1K comes from an unofficial mirror** (Hugging Face infinity1096/HD1K-Backup, commit
f07fd758, sha256 b599f378…) because the official download is an e-mail form; the user submits
the form to cross-check the bytes later [data/manifest/bundles.v1.json HD1K.zip disclosure;
reports/PLAN.md 2026-09-25 decision 3]. Every HD1K number discloses the mirror.

**L12. KITTI raw frames are re-encoded as JPEG** (Pillow 12.3.0, quality 92, 4:2:0; the
Monodepth2 README recipe) for transfer [bundles.v1.json conversions.jpeg-v1]. The U²Flow authors'
own JPEG quality is [U]. What it can bias: training inputs differ slightly from the authors'.

**L13. 23,604 training pairs, not the plan's 23,670** [A4; AGENTS.md section 7], re-measured on
the prepared data before prereg-v1. KITTI2015_MV_Sem reads 'training/Clean', which the official
multi-view layout lacks [V 152bb38:core/flow_datasets.py:306]; the codebase port changes the path, and
what the authors' directory held is [U].

**L14. The devkit carries sample GT of training id 000006** (devkit/matlab/data/) [M CRC-32,
2026-09-25]. In split v1 that id is dev [M], so the devkit is not sealed; u2f-meta-v1 excludes
the directory anyway.

## Compute and environment

**L15. All GPU work runs on one GPU type (Colab G4, RTX PRO 6000 Blackwell, sm_120) with torch
2.11**, not the authors' environment; the GATE-0 reproduction is what licenses comparing our G4
numbers with the paper's. The proposal's torch 1.13.1 fallback cannot run on sm_120 [V month
plan]. Every number records gpu_name and lock_hash (metric skill section 7).

**L16. Packing.** Up to two training processes share one G4 (config/ops.json packing); packing is
part of the protocol key, paired arms are packed alike or the difference is recorded, and σ_seed
includes packing noise.

<!-- L17-L44 appended at prereg-v2 from prereg/drafts/limitations-v2-additions.md (EV-0612) -->

## Regions (regions_v2, the R4 split)

**L17. R4s means "indistinguishable from the ego-motion at the KITTI outlier rule", not
"parked".** The KITTI-15 devkit says its foreground objects come from "fitting 3D CAD models to
individually moving cars" [V u2f-meta-v1 kitti15/devkit/readme.txt:11-13], and Menze & Geiger
annotate "all vehicles in motion" [sourced-V https://openaccess.thecvf.com/content_cvpr_2015/
papers/Menze_Object_Scene_Flow_2015_CVPR_paper.pdf, fetched 2026-10-03, sha256 a1955d1a83f4,
abstract]. regions_r4.py calls an instance moving when the median of |F_gt - F_ego| exceeds
3 px AND 5 % (the benchmark's own outlier rule, Menze & Geiger section 5). On the 100 dev ids
65 of 223 instances and 10.2 % of R4 pixels fall under that rule (static), 156 instances and
89.8 % are moving, 2 instances (11 px) are undetermined [M 2026-10-03, `regions_r4.py sanity`].
What it can bias: a slow mover counts as R4s; a rigid teacher is right there by the outlier rule,
so the H4 R4m margin is unaffected, but the word "static" in the paper must say this.

**L18. R4m-epi is empty on stationary-camera frames.** When the fitted ego translation is under
5 cm per frame the epipolar geometry is degenerate and no instance is R4m-epi. On the dev half
the PnP fit returns |t| = 0 with a 0.0 px residual on 15 of 100 frames, and |t| = 0.046 m on one
more (16 flagged) [M 2026-10-03]. What it can bias: R4m-epi under-represents scenes where the
camera waits at a light; it is a diagnostic region, never confirmatory.

**L19. The R4 constants were set here, before any R4 number existed.** MIN_INST_PX 50,
EGO_MIN_PTS 500, EGO_MIN_INLIER_FRAC 0.5, EGO_RANSAC_PX 1.0, EGO_RANSAC_ITERS 2000,
EGO_RANSAC_CONF 0.9999, EGO_SEED 20261002, EPI_MIN_T_M 0.05, and the reading of "5 %" as the
per-pixel relative residual, were pinned in tools/eval/regions_r4.py (file sha256 c27b8da5d78a at
2026-10-03T20:09:41+07:00, staged at tools/staging/eval/) before its first run on dev GT [M].
The dev run printed R4 counts and ego-fit residuals only (GT-only; no prediction, no R2 count).
The background ego fit is near exact (median residual 0.028 px, max 0.063 px over 100 frames;
inlier share 1.0) [M], consistent with the GT's construction from ego-motion-compensated laser
scans (Menze & Geiger section 4.1, same source). What it can bias: the constants are [E]; none
was tuned on an outcome; the SOTA checks of MIN_INST_PX and EGO_MIN_INLIER_FRAC are pending
(method skill thresholds list, two of its 7 pending items on 2026-10-06 [M `idea.py
sota-pending`]). EGO_RANSAC_ITERS and EGO_RANSAC_CONF are implementation parameters of the
RANSAC solver, not thresholds: with EGO_SEED fixed the fit is deterministic, they only bound
the solver's convergence, and they are not in the method skill's thresholds list (decision
2026-10-06, EV-0288).

**L20. The R4 split depends on the OpenCV build.** cv2 is 4.14.0 on the G4 [V env/lock_G0.txt]
and 5.0.0 on the Mac [M]; RANSAC results can differ between versions [A4]. GATE-1 G1.P4 requires
the instance classes and pixel counts to be identical on both hosts for the 100 dev ids before
any R4 row decides anything.

**L21. R4 covers CAD-fitted vehicles only.** Menze & Geiger manually mask "non-rigidly moving
objects like pedestrians or bicyclists" (section 4.2, same source), so those pixels are not in
V_occ and no region scores them. What it can bias: the "moving object" claims of the paper are
claims about vehicles.

**L22. R3-rel's 100 px is not the paper's number.** Menze & Geiger report that "very large
motions at the image boundaries (up to 500 pixels) degrade the accuracy of the ground truth",
and that "for most parts our ground truth is at least 3 pixels accurate" (section 4.2, same
source). The 100 px cut (regions.py R3_REL_MAX_PX, the proposal's) is therefore a choice inside
that range, not a published bound; GATE-1 reuses it for "reliable-GT pixels". Its SOTA check is
open (method thresholds.json R3_REL_MAX_PX).

## GATE-1 and PRIMARY v2 choices (each [E], each listed for the method skill)

**L23. Training-like windows are pooled 50/50.** A teacher's GATE-1 number pools the counts of
windows w03-11 and w10-18, because the training pairs sit one at each window end. The
self-triangulation teacher has no w10-18 (it needs t-1) [V tools/p1/self_triangulate.py header],
so it has no training-like number at all: it is compared only on w03-11, with the other
teachers' w03-11 rows (GATE-1 G1.F second check, diagnostic G1.a.selftri). Which way a
single-window comparison leans is not known in advance (the w03-11 row of a teacher may be
better or worse than its pooled row), so no direction is claimed. What it can bias: the
foundation claim against self-triangulation rests on one window position, the 3-11 end.

**L24. A teacher pixel without a value is an outlier (fail-closed).** A sparse teacher cannot
win GATE-1 (a) by abstaining; its coverage is reported beside it.

**L25. GATE-1 (d) reads delta-hat at eta = 0.55,** the midpoint of the plan's prior [0.4, 0.7];
the 0.4 and 0.7 values are recorded beside it, and the pilots replace eta before prereg-v3
(rule: L38).

**L26. Not every threshold has a literature check.** GATE-0's tolerances and the proposal's
GATE-1 / PRIMARY numbers were set from the paper and the plan [P]. On 2026-10-06 the method
skill's thresholds list holds 49 entries, 46 SOTA checks are recorded and 7 entries are pending (GATE-1
b, d and the foundation margin, the law's rho 0.7, Frig.b_max_px, R4.MIN_INST_PX,
R4.EGO_MIN_INLIER_FRAC) [M `tools/idea.py sota-pending`]. The values introduced by the
2026-10-06 decisions (eta floor 0.4 and cap 1.0, the 0.1 % count tolerance of G1.P4) are
hand-set and have no literature value to check against [E]. Re-run `sota-pending` at the tag:
the output, not this paragraph, is the record.

## Teachers (Tier 2 integrator, 2026-10-03; the record: tools/teacher/teachers.json and
## prereg/drafts/teachers.v2-draft.json)

**L27. Some compared teachers saw KITTI-like data.** pi3 is initialised from VGGT, which trained
on Virtual KITTI 2, whose Scene18 overlaps the KITTI scene-flow training scenes [P proposal;
V arXiv 2503.11651, 2507.13347]; pi3 also uses an undisclosed internal dataset [U]. Monodepth2's
released weights trained on Eigen-Zhou raw frames behind 72 of the 200 KITTI-15 training pairs
[M weights.json data_flag D]. Self-triangulation inherits the student's KITTI training. RAFT C+T
is supervised on synthetic flow (FlyingChairs and FlyingThings3D) and is recorded clean,
kitti_overlap no, in teachers.v2-draft.json (it is a synthetic-supervised flow prior, not a
rigid teacher). pi3, Monodepth2 and self-triangulation are recorded kitti_overlap yes, so a row
that uses them carries the flag; MapAnything (clean, no) and COLMAP (clean, no) are not flagged;
MapAnything's published data has no KITTI or Virtual KITTI but includes Spring. The verdicts
live in one registry (tools/teacher/teachers.json, sha256 pinned in teachers.v2-draft.json and
frozen by the tag); no idea.py teardown row exists for any teacher yet (the ideas ledger holds
no idea to attach one to, 2026-10-06 [M `idea.py list`]), as for SAM in L34.

**L28. Two KITTI-12 evaluation windows are truncated.** Sequences 000031 and 000082 start at
frame 05, so their 3-11 window holds frames 05-11; they stay in the 788 windows, flagged, and a
table that includes them reports the row without them beside it.

**L29. MapAnything's own intrinsics differ from the calibration.** On 3 dev frames its returned
fy was 256 against the calibrated 296 at 518x154 (up to 14 %) [M teacher-factory, CPU]; both K
and K_calib are cached and the one that renders F_rig is fixed per teacher at W2-14, before any
G1 number: render_frig.py and refine_pose.py refuse a predicted-K cache without an explicit
`--k teacher|calib` and record the choice (GATE-1 _open).

**L30. One K per window.** The proposal's rigid flow has one K (equation 0). A teacher that
predicts or fits K per frame (MapAnything, pi3) is rendered with the window's one K (frame 10's
own, or the calibration), never frame 10's K to back-project and frame 11's to project: with an
identity pose a 1 % K difference between the frames alone made 6.5 px of flow at native
resolution [M tests/test_p1_rigid.py]. What it can bias: nothing the gate reads; the
alternative was a spurious flow growing toward the border, i.e. in R3.

**L31. Teacher rows are rendered on the Mac and scored like the student.** A teacher's GATE-1
row is F_rig on the 256x832 training grid carried to native resolution by U2Flow's own
resize_flow (the student's path; the direct native render is a diagnostic only). Its gpu_name is
the GPU that ran the teacher's network (or, for self-triangulation, U2Flow's evidence); the
render (numpy float64) and the PnP of the refined pose (cv2 5.0.0) run on the Mac, recorded
beside it. Training-window refined poses made later on the VM use cv2 4.14.0; their parity with
the Mac's is not yet checked.

## C0 as registered (EXP-005; critic review of EXP-004/EXP-005, 2026-10-04)

**L32. The unlabelled training stream is not restricted to the dev half (qualifies L5).** C0's
stream is KITTI2015_MV_Sem: every consecutive frame pair of image_2 and image_3 in KITTI-15
multi_view testing and training except pairs touching frames 9-12, with no dev/held-out filter;
only the GT is gated [V core/flow_datasets.py:295-332; mixes KITTI-12 as well, :377-390, critic
EV-0149 and the second review]. So the photometric loss and the SAM masks see the neighbouring
frames of the held-out half's scenes, and never the evaluated pairs (frames 9-12). L5 says
"HO-tr removes our own training signal on held-out scenes": that holds for what a teacher or a
pseudo-label adds (gated by the held-out id list), NOT for this stream unless the loader hook
(W4-03) filters it; whether it should is a prereg-v2 decision for the user. What it can bias: the
absolute level of every arm on the held-out half equally (C0, PF and M share the stream); H4's
contrast is not confounded. Evidence: reports/experiments/EXP-005.md, EV-0149.

**L33. sigma_seed is a same-packing quantity (corrects L16's last clause for C0).** The three C0
seeds run solo, one per G4, so their sigma_seed contains no packing noise; L16 ("sigma_seed
includes packing noise") holds only for runs that actually share a G4. PF and M seeds that run
packed are compared with C0 solo on a different packing: the kill and the primary waive `packing`
by design (the plan: packing changes speed, not the computation), but a seed-to-seed difference
that depends on packing would enter the paired contrast. What it can bias: sigma_seed (low if
packing adds variance); the paired contrast is matched by seed index, not by packing. Evidence:
EV-0157; packed runs are g0-seed-s1/s2 only.

**L34. The SAM masks are a teacher cache that was never audited for KITTI overlap.** The sam:v1
masks (ViT-H, UnSAMFlow's recipe) cover all 26,760 multi-view frames, held-out scenes included,
and C0's stage-3 semantic augmentation consumes them. SAM belongs to the authors' recipe, so every
arm shares it and it cannot bias an arm contrast; whether SAM's own training set (SA-1B) overlaps
KITTI is [U], and no teardown (tools/idea.py teardown) exists yet. What it can bias: the absolute
level, equally for all arms. Evidence: EV-0158; to do: a teardown row before prereg-v2.

## Decisions of 2026-10-06 (the record: reports/decisions_2026-10-06.md; each [E] unless marked)

**L35. HO-tr is one list: all 200 KITTI-15 training sequences, and no KITTI-12 sequence has a
teacher.** The plan (dev half off), the proposal (all 200 off) and the loader code (held-out half
off) disagreed; the proposal's list is the only one under which a pilot scored on the dev half
and a confirmatory run scored on the held-out half are trained alike (prereg/ho-tr-teacher-off.json,
sha256 in PRIMARY protocols.HO-tr). KITTI-12 pairs carry no teacher (L_FF only), so the teacher
supervises KITTI-15 testing sequences only; that is about a quarter of the 3,156 teacher windows
[E from the make_windows.py docstring: 200 sequences x 2 windows x 2 cameras = 800 of 3,156; the exact count is U until `make_windows.py --set train` runs on the data].
What it can bias: the teacher's share of the batch is smaller than the proposal's, which can only
shrink the M-PF difference (conservative for H4). KITTI-15 testing sequences may come from raw
drives that also hold held-out scenes; the split maps drives for the 200 training sequences only
(L3), so a held-out scene can have a teacher-supervised neighbour in the testing set. HO-all is
not run (it would leave no teacher at all).

**L36. The H4 control is PF without FIW.** PF = C0 + L_FF with beta = 1 (the online-student
visibility factor is dropped, the offline masks are baked into the cache) and no full-image
warping. The proposal's PF includes FIW. The plan runs the pilots FIW-less, and FIW lands in month
2, after the pilots. What it can bias: a weaker control than the registered PF makes the M-PF gap
larger than against the full PF if FIW and the teacher overlap in what they fix (both help
out-of-frame pixels); eta_hat comes from the same control, so condition C2 stays consistent, but
the claim is restricted to "on top of FF self-distillation without FIW". PF-FIW is an exploratory
arm (batch B), never the control. The paper says "PF (no FIW)" wherever PF appears.

**L37. M is not the proposal's M.** L_T is forward only (d = 12) at weight 1 instead of 1/2 per
direction over two directions; the backward l_hg keeps RIGID segments (no frame-2 RIGID map);
L_T applies where w > 0 AND RIGID (so M with w_T = 0 is the F1 control, not PF); beta = 1; no
teacher on KITTI-12. Inference uses the forward head only, so the missing backward supervision
and the backward smoothness over rigid segments most likely cost M accuracy rather than give it
some [E, not measured]. Evidence: reports/cycles/CYC-004.codereview.json user_decisions 1-6.

**L38. GATE-1 (c) and delta-hat are dev-in-sample; eta_hat is a one-seed estimate.** The gate
constants and the arbitration gate are tuned on the same 100 dev images that (b), (c) and (d)
read, so those criteria are optimistic about the held-out half; the constants in GATE-1.json are
frozen by the prereg-v2 tag before any composite row exists. delta_hat1 is a signed count
difference on the support C (w > 0 AND RIGID pixels of R2 u R3), not the proposal's per-region
positive part; eta_hat is one seed (seed 1, 10k steps, the plan's pilot), its image-bootstrap CI
carries no seed noise, and eta_used is clipped to [0.4, 1.0] (PRIMARY delta_hat_fraction.eta).
What it can bias: a 10k pilot understates a 30k run (conservative for the 50 % condition only if
eta grows with steps, which is [U]); transporting dev counts to the held-out half has no error
bar.

**L39. pi3 is a diagnostic, never the headline, in prereg-v2.** pi3 is flagged V (VGGT, Virtual
KITTI 2 Scene18 overlapping KITTI scene-flow training scenes, L27) and has an undisclosed
internal dataset [U], so a 1-point margin on dev cannot make it the headline. GATE-1 (a), (b), (c),
(d) and G1.F decide on MapAnything; pi3's numbers sit beside them (G1.T, G1.a.pi3, G1.F.pi3).

**L40. Held-out R4 is computed on the Mac only.** cv2 is 5.0.0 on the Mac and 4.14.0 on the G4;
G1.P4 shows R4 parity on the 100 dev ids only. The held-out half's GT sits sealed on the Mac, so
its R4 split cannot be cross-checked on a VM (rule 4). What it can bias: if the two builds split
an instance differently, R4m on held-out could differ from the dev-validated one by an amount
that cannot be measured.

**L41. H4 may be infeasible, and delta_hat/eta are not predictions.** (a) delta_hat1 is a
counterfactual: under the HO-tr list the teacher is off on every dev and held-out image, so the
supports C that delta_hat1 measures are never supervised in M; eta = g / delta_hat1 is a
cross-scene transfer ratio from KITTI-15 testing windows only, and its prior [0.4, 0.7] has no
literature value and no ledger measurement (an exploratory choice [E]; PRIMARY
delta_hat_fraction.eta). (b) The effect H4 needs (0.2 pp, about 23% of the 0.86 pp ceiling) is
reached, if at all, by transfer alone; sigma_seed is 0.036 pp [M, RES-0039/0069/0099], so noise
is not the constraint but the effect size is, and the only evidence of it (one 10k pilot) arrives
after the prereg-v2 tag. PRIMARY feasibility.pilot_gate therefore stops the confirmatory runs when
the pilot's dev gain is below 0.10 pp; H4 is then reported as not tested, not as failed. (c) C2
(50% of delta_hat_used) binds only when eta_used * delta_hat1 > 0.4; below it C1 (the 0.2 floor)
decides (PRIMARY conjunction.binding_regime). (d) G1.d screens a PF proxy (U2Flow+FF), not the
control H4 tests; v3 recomputes on the PF pilot. (e) The eval_sha pin is written only after the
last edit of a hashed file (reports/eval_round_v2.md 1.5), the analysis code lives in
tools/prereg/prereg_stats.py outside the hashed set (EV-0309).

**L42. The PF and M pilots train on a declared 50 % subset of the sequences, not on the full
stream (decisions_2026-10-06.md, Round 3 (compute)).** Reason: the FF pseudo-label job runs at
0.36 items/s [M, p1-ff-train-r3 bus], so the 20,448-pair list needs about 15.8 h of one G4 and the
pilots need their FF cache, the teacher cache and the gate maps on ONE VM (user decision 5: caches
never leave it). The list is prereg/drafts/pilot-subset.v2-draft.json (sha256 99aadac2...43f5b,
table_sha256 b2c1cea4...288e): 395 of 789 sequences (both cameras), 11,818 of 23,604 loader pairs,
seeded (20261006, data-independent), drive-balanced to within one sequence per drive, the same
fraction in each of the four strata (K15 training split by its dev and held-out halves, K15
testing, K12 training, K12 testing), so the stream keeps C0's composition. The same list filters
the trainer's dataset for PF and for M and selects the FF items and the teacher windows; both arms
read it, the confirmatory runs (30k steps, 3 seeds) train on the full stream. What it can bias:
(a) eta_hat is measured on a pilot that sees half the sequences (about 3.4 passes over 11.8k pairs
instead of 1.7 over 23.6k) and half the teacher-on scenes (100 instead of 200 KITTI-15 testing
sequences), while H4 tests the full-stream runs; eta transports across both steps and scenes with
no error bar. The direction is [U]: fewer teacher scenes may understate the transfer (a lower
eta_hat loosens C2, bounded below by the floor 0.4 and by C1's 0.2 pp floor, which does not depend
on eta), repeated passes over fewer scenes may overstate it. (b) The drive proxy 'blk<id // 10>'
for the sequences the split does not map to a drive (K15 testing, K12, 58 of 200 K15 training
ids) is [U]. (c) The 10k-step pilot samples 84 % of the 23,604 pairs on the full stream [E] and
98 % of the subset, so the subset costs coverage of scenes, not of steps. (d) GATE-1 G1.P3 as
drafted reads `complete` of the FF manifest for the full 20,448-pair list; a subset manifest says
`complete: false` and `complete_selection: true` with the selection's table_sha256, so the FF
clause of G1.P3 is amended to that (draft edit, not yet applied: GATE-1.json G1.P3 second check,
fields data.complete_selection == true and data.selection.table_sha256 == the table_sha256
above); the MapAnything clause stays as drafted (the full 3,156-window cache is complete,
manifest files_sha256 62bd93da...198a [M]).

**L43. FF labels in the pilots cover KITTI-15 pairs only and the forward direction only.** The FF
cache of the pilots is built for the KITTI-15 sequences of the subset (5,188 items); KITTI-12
pairs train photometrically in both arms (core/u2f_providers.py include_k12=False), and the cache
holds the forward direction only, so under the loader's temporal swap about half of the K15 samples
carry no FF label (allow_swap_zero, decision 2026-10-06 of the PF arm). PF in the pilots is
therefore a narrower L_FF than the proposal's (all pairs, both directions); M adds L_T on top of
the same labels, so the M-PF contrast is unchanged but its absolute level is not the proposal's PF.
The confirmatory PF and M must use the same labels (a full-stream K15 cache, 10,378 items, about
8.0 h at the measured rate, to be built on the confirmatory VM).

**L44. What the M pilot's teacher weight is (gate maps, tools/p1/gate_maps.py; added 2026-10-06).**
(a) w = the A8 gate weight (min(1, n_s/200) x pair_w, zero within 3 px of a non-RIGID segment and in
R3 outlier carve-outs, zero where a fold pose failed) x valid_T (the rigid render is defined: depth
valid, no front veto, b_T <= b_max). Out-of-frame (OOF) and teacher-occluded (O_T) pixels of an
accepted segment KEEP the segment's weight on purpose: the claim is "teach where photometry is
blind", the proposal's beta selects exactly those pixels, and the estimand delta_hat1 and the H4
support are defined on R2 union R3, which on the dev half is in practice inside R3 (occluded and
out of frame; R2 has no GT-valid pixels there, L41). beta = 1 holds for FF (L36/L37); the teacher is
applied on occluded and OOF pixels by design. The evidence-side filters (e^alpha <= 2, visible,
valid_T, not OOF, not O_T) only decide which pixels give evidence for the pose fit and the segment
statistics, not which pixels are supervised. (b) Not applied: c_T (the confidence-rank factor of
the proposal's w; the teacher's own confidence is unused), A2's sky and depth-edge tests, and R1's
relaxed evidence mask; the manifest says so (settings.not_applied). (c) F_T and b_T are rendered
with T_ref, the PnP pose on the teacher's own depth, with the calibration K (the W2-14
recommendation, still open in GATE-1); A3's pair_w (PnP failure, inlier ratio < 0.5, < 300 inliers,
T_ref off the teacher pose by > 0.5 deg or 10 deg) zeroes a whole frame, and the rate is read from
the run's gate_maps_summary.json (pair_w_0_by_class) before the pilot, [U] until then. (d) The SAM
tree is the pack's two-level tree (full_seg labels and key objects), not the AMG hierarchy of A1;
the pilot spec uses --tree sam, so a frame without a mask fails instead of falling back to the 32 x
32 grid tree. (e) Pose disagreement between the two fold poses is counted per pair and not
withheld (--disagree report, the reviewed library's behaviour); the CPU stage can be rerun with
withhold. (f) The maps are forward-only and RIGID is segment-level (the gate's accepted segments),
so under the temporal swap L_T acts on about half of the teacher-on samples (L43, allow_swap_zero).
(g) The teacher acts on the KITTI-15 testing sequences of the subset only (HO-tr, decision A); the
producer, the provider and the trainer setup each refuse a K15 training sequence that is not on the
teacher-off list.

## Additions for prereg-v2/v3 (ARB primary endpoint, drafted 2026-10-12)

Numbered from L45 so they cannot collide with limitations-v2-additions.md (L17-L44).

**L45. The primary endpoint changed from H4 to ARB before anything held-out was seen.** H4
(M vs PF at HO-tr) cannot be tested: the M' pilot was NO-GO under a rule declared before its
result (EV-0447, EV-0502), so no trained arm exists. ARB tests a different, inference-time
procedure. The paper reports H4 as not tested, never as negative.

**L46. GATE-1.json is a superseded placeholder.** prereg/GATE-1.json (from
prereg/drafts/GATE-1.superseded.json) holds one placeholder criterion, due 2099-12-31, and no
decision rule; it exists only because tag.sh requires the file for prereg-v2. The H4-path
criteria it replaced were closed by EV-0502, and their dev numbers already existed (EV-0301,
EV-0401). It is never evaluated or signed off (EV-0594) and supports no claim.

**L47. The rule's thresholds were chosen in-sample on the dev half** (α > 2.75, |T−S| ≤ 100 px;
EV-0402, 0.25 grid). They were frozen before KITTI-12 (EV-0403) and are unchanged here. Dev
numbers of the rule are optimistic. The held-out half is held out from the rule's tuning and from
every view of its GT, not from kitti.pth's unsupervised training, which saw neighbouring frames of
every KITTI-15 training scene (L5); the protocol is therefore FULL (critic EV-0572).

**L48. The baseline levels of rel and +FF on the held-out half are derivable.** The GATE-0
all-200 aggregates and the dev rows determine them exactly: the rows carry pixel counts
(RES-0032, RES-0034/0036/0038 all-200; RES-0001 dev). They were never computed. The endpoint
choice (rel as primary, +FF as S1/S2) rests on the KITTI-12 results EV-0403 and EV-0406, not on
any held-out level. After unseal the scorer derives the rel level from those rows and requires the
scored level to equal it (PRIMARY validity_checks.rel_level, a GT-integrity check; a mismatch makes
the run invalid). The arms arb and ffarb have no held-out number of any kind before unseal.

**L49. The teacher is not independent of S.** selftri is a three-frame rigid refit of
kitti.pth's own 10→09 and 10→11 flows (tools/p1/self_triangulate.py). The procedure uses
frames 09, 10 and 11 and costs a second forward pass plus pose and DLT triangulation per image.
It is compared with +FF, which is also three-frame and per-image optimised, and never at
U²Flow's two-frame inference cost.

**L50. Two held-out windows have no selftri item**: 000091 and 000094, where recoverPose found 0
points in front (U2FLOW_data/runs/p1-selftri-r1/out/cache/diag/selftri/failures.csv). They keep
R = S and stay in the denominator. The cache's item list is pinned (prereg/arb_data_manifest.json),
so no later run fills them. The rule was tuned on dev with 96/100 windows (000043, 000046, 000113,
000114 missing for the same reason, same handling).

**L51. The +FF producer differs from IDEA-005.** On KITTI-12, IDEA-005 used the p1-ff-eval-r3
cache (FF seed 1). Here +FF is the GATE-0 predictions EXP-003-ff-s1/s2/s3 (seeds 1-3, the
G0.3 reproduction). +FF is not deterministic per item (EV-0406, item 000082), hence three seeds
and intersection-union decisions.

**L52. KITTI-15 split limits apply** (L3): 58 unmapped sequences are not scene-disjoint across
halves. The image is the bootstrap unit.

**L53. S comes from a different producer than on KITTI-12.** Here S is the GATE-0 prediction set
EXP-002-audit-s1 (eval_sha e79cd87edc2f); IDEA-004 read native_10to11 from p1-dump-eval-u2flow-r1.
On dev the two are bit-identical (dump_eval.py compare, max |diff| 0.0 px over 100 ids). The
dump holds no native flow for held-out ids, so a GT-free check compared EXP-002's predictions with
resize_flow of the dump's fp16 grid flow on both halves: the residual is the same on dev and
held-out (mean 0.00185 vs 0.00222 px, max 0.176 vs 0.093 px; EV-0591). S comes from the
evaluator pass and alpha and T from the dump's fp16 pass (EV-0613); the arrangement is the same on
both halves, which is what the comparison needs.

**L54. The +FF arm on dev reproduces S2 at 0.335 pp, not EV-0406's 0.320 pp** (seed 1, in-sample),
because the FF arm here is GATE-0's EXP-003-ff-s1, not the p1-ff-eval-r3 cache IDEA-005 used
(L51). Recorded by the scorer's reproduction check (commit 7e9535c).

**L55. Held-out T is rendered by tools/p1/render_heldout.py**, which calls the render function of
render_frig.py; render_frig.py itself is unchanged and still refuses held-out ids. The new script
re-rendered the 96 dev windows bit-identical to the KITTI-12/dev render, then rendered the 98
held-out windows from images only (no GT). Its output's pred_sha is pinned in inputs_sha256.T.

**L56. Multiplicity departs from statistics.md.** S1 and S2 use Bonferroni (97.5 % CIs each), not
Holm: the decisions read confidence intervals, and a step-down method is not built in the scorer.
Bonferroni intervals are conservative against Holm. The intersection-union over FF seeds inside S1
and S2 needs no further correction (critic EV-0578).

**L57. Batch B stays sealed in this phase.** The held-out scorer builds batch A rows only (Fl, all,
px-pooled, the four arms and their contrasts); R1, R3, occ Fl, EPE, switch shares and per-image
signs are not computed (critic EV-0573).

Earlier agent self-review (2026-10-12, superseded by the u2f-critic review EV-0568..0579; the u2f-critic agent could not be spawned from this worker): no
blocker found. One major was fixed: the frozen file must not contain null in
primary_endpoint (unseal.py _nulls), so d_gt_px became "none". Two minors were addressed: L53
(producer of S), and the Bonferroni levels made explicit. Open for the critic before prereg-v3:
whether S1/S2 should use the seed-mean +FF instead of intersection-union. IUT is the more
conservative choice and is kept.

**L58. GATE-1 is superseded, not evaluated (EV-0594, EV-0611).** prereg/GATE-1.json is a minimal
placeholder so that prereg-v2 can be tagged; `gate.py due` never lists it. A `gate.py eval GATE-1`
or report.py dry run would print GO once prereg-v2 exists: that output means nothing and is never
quoted.

**L59. L17-L44 are carried over unchanged (EV-0612).** They were written for the v2 PRIMARY
(prereg/drafts/limitations-v2-additions.md) and are appended with L45-L59 so the numbering has no
gap. For ARB, L28 (two truncated KITTI-12 windows), L30 (one K per window) and L32 (the
unlabelled stream was not restricted to dev, qualifying L5) apply; the H4/pilot ones do not.
