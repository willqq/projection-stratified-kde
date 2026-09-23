# Reproducing the frozen ICASSP sprint

This delivery preserves the scientific source at commit `59d86dbfaaeebb7bdaace151afcec047f3e1a360`, branch `icassp-sprint-20260916`. The starting commit was `296379cb3c912f309bd49d97046067d191f9ff24`. The frozen method is the shared 64-dimensional Gaussian projection, eight quantile strata, exact score-selected H of size floor(M/4), complete reference support and proportional allocation. The original feature-space Gaussian kernel and data splits are unchanged.

Packaging did not rerun experiments or modify the scientific HEAD. The Git bundle was cloned locally and its HEAD verified. All 332 files in the code/assets archive were checked against their SHA-256 manifest. The four raw data files were read only to verify their published delivery hashes. Pattern-based credential screening reported no matches; `SECRET_SCAN.json` records its scope and limits without matched content.

## What is included

| Artifact | Purpose |
|---|---|
| `reproducibility/frozen_branch.bundle` | Self-contained frozen Git branch, including its reachable history; preserves the HEAD required by the formal evaluator. |
| `reproducibility/from_296379_to_59d86db.patch` | Binary-safe diff against the actual starting commit. |
| `reproducibility/code_assets.tar.gz` | Frozen tracked source, the Linux posting extension, legacy MECH/preprocessing/split assets, projection/calibration caches, official DEANN source and built extension, audit IDs, intake, logs and aggregation script. It also includes the small prior-final config and validation CSVs needed by `sprint_smoke.py` and `sprint_freeze.py`. |
| `reproducibility/independent_audit/sprint_audit_results.py` | Later independent result checker. It is intentionally separate from the frozen scientific commit. Its SHA-256 is recorded in `git_state.json` and the delivery manifest. |
| `reproducibility/audit_reference_inputs.tar.gz` | Previous formal baseline CSVs and the finite-population estimator-check summary needed by the independent audit. These are saved experiment outputs, not raw datasets. |
| `reproducibility/SOURCE_ASSET_MANIFEST.json` | File-level hashes inside the archive. |
| `reproducibility/environment_hardware.json` | Captured Python/package versions, CPU, memory, compiler, CMake and GPU inventory. |
| `reproducibility/formal_launch_environment.json` | Actual formal-launch thread and library settings, confirmed by the executing agent. |
| `reproducibility/DATA_MANIFEST.json` | Raw data relative paths, server locations, byte sizes and verified SHA-256 values. |
| `audit/`, `models/`, `intake/`, `logs/` | Local copies merged without overwriting different existing files; merge record is `INTAKE_COPY_MANIFEST.json`. |

The raw datasets, virtual environment, credentials, SSH configuration and shell history are excluded. `models/projection64` contains the frozen derived 64-dimensional reference projections as well as the Gaussian matrix and training mean. Original d-dimensional features still have to be supplied from the four matching raw files. Results are delivered separately under `results/` and `analysis/`.

## Restore an isolated replay directory

The following shell commands describe a new Linux replay directory. Do not run them over the delivered results or the original experiment directory. Replace `DELIVERY` with the path where this delivery is available on that machine. These commands are documented for author replay; packaging executed only help, metadata and integrity checks.

```sh
DELIVERY=/path/to/sprint_final
SPRINT_ROOT=/root/autodl-tmp/icassp_sprint_replay
mkdir -p "$SPRINT_ROOT"
git clone --branch icassp-sprint-20260916 \
  "$DELIVERY/reproducibility/frozen_branch.bundle" "$SPRINT_ROOT/worktree"
tar -xzf "$DELIVERY/reproducibility/code_assets.tar.gz" -C "$SPRINT_ROOT"
tar -xzf "$DELIVERY/reproducibility/audit_reference_inputs.tar.gz" -C "$SPRINT_ROOT"
git -C "$SPRINT_ROOT/worktree" rev-parse HEAD
git -C "$SPRINT_ROOT/worktree" diff --exit-code
```

The last hash must equal `59d86dbfaaeebb7bdaace151afcec047f3e1a360`. Extraction adds the compiled posting extension as an untracked file; it leaves tracked source unchanged. The source archive alone has no `.git`, so use the bundle when executing the frozen formal evaluator.

Place the raw files beneath `$SPRINT_ROOT/worktree/data/` with these exact relative paths, or link that directory to an existing verified data root:

| Dataset | Relative path | SHA-256 |
|---|---|---|
| ISOLET | `isolet/isolet1+2+3+4.data` | `aa935cc98ec22ee689aa5b2ddf1fdce074692f7016f0255f0982df133ccbab38` |
| CIFAR-10-Small | `cifar-10-batches-py/data_batch_1` | `54636561a3ce25bd3e19253c6b0d8538147b0ae398331ac4a2d86c6d987368cd` |
| GIST-512 | `cifar10-Gist512/Cifar10-Gist512.mat` | `0d55fa328e8fe5a262498a5967f01f91d0ab5e053ddbbe49e473609a22cf5aff` |
| Amazon | `Amazon_initial_50_30_10000/Amazon_initial_50_30_10000.arff` | `fd757a3517a850e5af8136c343fde492dd80f02af1e3c8e995b6da1154324aa5` |

The source loaders enforce these hashes. This package does not claim to redistribute or reacquire those datasets. `assets/*/split_ids.json` preserves the original split, and `audit/locked_ids.json` preserves the independent audit IDs. There are 200 audit queries for each of the first three datasets and none for Amazon.

## Runtime and native dependencies

The measured host used CPython 3.8.10 on Linux x86-64, Intel Xeon Gold 6459C CPUs, eight compute threads, NumPy 1.24.2, SciPy 1.10.1, PyTorch 2.0.0+cu118 and FAISS CPU 1.7.4. CUDA was disabled for formal experiments. The GPU inventory describes the host and does not imply GPU evaluation. The runtime used a virtual environment with access to installed base packages; the full package inventory is a capture, not a portable lockfile.

`requirements.runtime.txt` lists the directly relevant installed versions. A fresh Python 3.8 environment can be prepared using these pins, with the PyTorch cu118 wheel index if needed. Installation has not been replayed on a clean machine in this delivery. Linux ABI compatibility and MKL availability must be checked before claiming a clean rebuild.

On the original host, the actual launch configuration was:

```sh
PY=/root/autodl-tmp/icassp_revision_20260915/.venv/bin/python
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=
export LD_LIBRARY_PATH=/root/autodl-tmp/icassp_revision_20260915/.venv/lib
REV="$SPRINT_ROOT/worktree/mech_annulus_project_clean/revision"
```

For another compatible host, set `PY` and `LD_LIBRARY_PATH` to its Python 3.8 runtime and MKL library directory. Preserve the four thread settings. The supplied DEANN extension requires MKL ILP64, MKL GNU threading and MKL core `.so.2` libraries; `deann_binary_linkage.txt` records its actual dependencies. The supplied extensions target CPython 3.8/Linux x86-64 and do not run natively on macOS.

The DEANN source is from official commit `6dfb28370d6ed50a0aef53ece111fa449503554e`; its estimator and official FAISS wrapper were independently hash-matched to upstream (see `literature/deann_source_check.json`). The local `deann_adapter.py` records the common-precision wrapper. DEANN's source/build files and license are preserved in `external/deann`. `build_posting_index.py` is the exact posting-extension build entry point.

If a rebuild is required, these are the source-supported commands in a separate environment:

```sh
"$PY" "$REV/build_posting_index.py"
cmake -S "$SPRINT_ROOT/external/deann" \
  -B "$SPRINT_ROOT/external/deann/build_rebuilt" \
  -DMKL_ROOT=/path/to/mkl-runtime \
  -DPYTHON_EXECUTABLE="$PY" \
  -Dpybind11_DIR="$("$PY" -m pybind11 --cmakedir)" \
  -DENABLE_MKL_MULTITHREADING=ON
cmake --build "$SPRINT_ROOT/external/deann/build_rebuilt" --parallel 8
```

DEANN's adapter searches `external/deann/build` and requires exactly one `deann*.so`. Keep the supplied build separately before installing the rebuilt output there. Rebuilding can change binary hashes even with identical source. The formal freeze checks the posting-extension hash; it will reject a different binary. A portable rebuilt run therefore needs a separately documented environment/freeze manifest. Preserve the delivered manifest and report such a run as a rebuild replication, rather than changing the original frozen record to make a guard pass.

## Replaying the already frozen formal evaluation

The archive contains the delivered config and freeze marker, but no formal result directories. Models are reused, not retrained. The commands below retain every old test query and every locked audit query. Run them serially and keep other training/experiments off the timing host.

```sh
mkdir -p "$SPRINT_ROOT/results" "$SPRINT_ROOT/qa"
for dataset in isolet cifar10 cifar10_gist512 amazon; do
  "$PY" "$REV/sprint_formal.py" --dataset "$dataset" --phase test
done
for dataset in isolet cifar10 cifar10_gist512; do
  "$PY" "$REV/sprint_formal.py" --dataset "$dataset" --phase audit
done
"$PY" "$REV/sprint_fixed_c_final.py" --phase test --dataset all
"$PY" "$REV/sprint_fixed_c_final.py" --phase audit --dataset all
"$PY" "$SPRINT_ROOT/scripts/summarize_sprint.py"
"$PY" "$DELIVERY/reproducibility/independent_audit/sprint_audit_results.py" \
  --root "$SPRINT_ROOT"
```

`sprint_fixed_c_final.py` explicitly skips Amazon audit when no locked rows exist. Formal evaluator output directories use `exist_ok=False`; this protects prior runs. Use a fresh replay root instead of deleting saved results. CLI help for all flagged scripts was actually executed during packaging and is saved as `*_help.txt`.

Budgets are 32/64/128/256/512 and sampling seeds 0–4. Each approximate method has 200 queries per available phase. Exact sum uses one observation per query. Basic methods reset the sampling RNG for each of three timing calls, using `seed + query_id*100003`; their first estimate supplies error. Official DEANN uses a warmup and three advancing permuted-sampler calls; the first supplies error and each call retains its actual work counts. Reported latency is the median of those three calls at the already-transformed-query boundary. Raw feature transform is measured separately. Do not sum phase medians as though they were a single measured total.

The frozen DEANN `(nprobe, near_fraction, nlist)` values are `(4,.5,32)` for ISOLET, `(4,.25,32)` for CIFAR, `(4,.25,32)` for GIST and `(4,.5,18)` for Amazon. Prior validation selection records are reflected in `final_config.yaml`. No DEANN retuning is part of formal replay.

## Reconstructing validation and freeze

This is optional provenance replay, distinct from checking the already frozen method. Use another fresh restored root, preserve the delivered `final_config.yaml` and `freeze_manifest.json` outside its active root, and leave its `results/validation` empty. Do not perform parameter selection in a directory that already contains formal test/audit results. Preserved models and audit IDs remain fixed.

The original bounded workflow uses these exact script interfaces:

```sh
mkdir -p "$SPRINT_ROOT/results/smoke" "$SPRINT_ROOT/results/estimator_checks"
"$PY" "$REV/test_sprint_estimator.py" --out "$SPRINT_ROOT/results/estimator_checks"
"$PY" "$REV/sprint_fixed_c_final.py" --self-test
"$PY" "$REV/sprint_smoke.py"
"$PY" "$REV/sprint_group_diagnostics.py" --dataset all
"$PY" "$REV/sprint_validate.py"
# B1 was activated after the recorded B0 diagnostic gate; keep that decision record.
"$PY" "$REV/sprint_b1_screen.py"
"$PY" "$REV/sprint_finalists.py"
"$PY" "$REV/sprint_control_refresh.py"
"$PY" "$REV/sprint_select.py"
"$PY" "$REV/sprint_freeze.py"
```

The scripts without arguments do not offer a safe `--help` entry point and were checked by reading source, without executing them during packaging. `sprint_smoke.py` uses the included prior-final validation CSVs. The B1 activation rationale is `intake/B1_ACTIVATION.md`. A pre-freeze matched-control ID-order correction is documented in `intake/MATCHED_CONTROL_ORDER.md`; old control results remain preserved in the separate results delivery. Fresh runs use the corrected frozen code throughout, so `sprint_control_refresh.py` can reuse its identical cached control.

`sprint_select.py` encodes the final recorded selection and asserts the component guard outcomes. It is a reproducibility check for this bounded study, not a generic search utility. If a fresh environment does not reproduce those guard outcomes, stop and report the discrepancy. Do not relax an assertion after inspecting test/audit data. `sprint_freeze.py` requires no existing freeze marker and writes time-dependent provenance; a newly generated manifest will not have the delivery's original hash or timestamp.

## Independent checking and limits

The later `sprint_audit_results.py` reads raw formal CSVs and frozen provenance without importing estimator modules or reading parent aggregate tables. It recomputes query-level summaries and 1,000 paired bootstrap replicates, averaging the five seeds within each query before resampling. Its code remains outside scientific HEAD. The independent checker reports its own source hash and explicitly records limits of the per-query fields.

Old test queries were already exposed during earlier research. The three locked audit sets provide additional independent evidence; Amazon has no unused rows and no independent audit claim. Bitwise estimates may depend on numerical library/ABI details, and latency depends on hardware and scheduling. The package preserves the original evidence and permits author verification; packaging alone does not certify a clean installation or a new execution of the experiments.
