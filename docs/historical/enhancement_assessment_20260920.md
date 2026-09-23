# Reproduction and provenance

The frozen scientific estimator is unchanged. This directory contains evidence checks and analysis, not a revised manuscript.

## Entry points

- `EXPERIMENT_DECISION.md`: pre-evaluation selection and protocol. Its SHA256 is in `baseline/MANIFEST.json` and every remote stage-start record.
- `EXPERIMENT_RESULTS.md`: completed results, including negative/uncertain conditions.
- `PAPER_UPDATE_DECISION.md`: C. CLAIM NARROWING; proposed minimal manuscript changes, not applied.
- `MECHANISM_EVIDENCE_AUDIT.md`: reuse of existing fixed-candidate controls and the limits of a recall/variance correlation claim.

## Code and environment

Scientific base: `59d86dbfaaeebb7bdaace151afcec047f3e1a360`.

Previous seed-check revision: `6741795be077b813cedd5ed888a5a692fb266b6e`.

New branch: `icassp-enhancement-20260920` at `/root/autodl-tmp/icassp_enhancement_20260920/worktree`.

Bandwidth/fairness driver and protocol commit: `5e5b492`. Scaling driver commit: `2501679`. The latter adds a start timestamp and removes inherited nlist metadata from a query-parameter dictionary; actual IVF construction remains `min(32, floor(n/40))`. Both executed scripts are retained under `results/`; their checksums agree with stage-start records. `results/enhancement.patch` contains the new work relative to 6741795. `source/` preserves the imported scientific Python modules. The complete scientific repository, native DEANN build and original model assets remain in the prior sprint's reproducibility bundle and unchanged server directories.

`results/ENVIRONMENT.json` records versions, CPU and repository states. Runs use the existing CPU environment with eight threads and no concurrent experiment/training processes. Every comparison begins with an already-transformed query, as in the frozen paper; raw preprocessing is outside that common estimator timing boundary. Projection/IVF build time and storage are recorded separately in each `offline.json`.

## Remote invocation

Use the configured SSH identity or interactive authentication. No credential is stored in these files. Preserve the existing host identity checks.

From the isolated remote root, the runs were launched sequentially with:

```sh
env OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
NUMEXPR_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= \
LD_LIBRARY_PATH=/root/autodl-tmp/icassp_revision_20260915/.venv/lib \
/root/autodl-tmp/icassp_revision_20260915/.venv/bin/python \
worktree/mech_annulus_project_clean/revision/enhance.py bandwidth
```

Then `deann_validation`, `deann_formal`, and `scaling` replaced the final argument. Stage output directories are exclusive: rerunning in place is intentionally rejected. Reproduce in a fresh isolated root with the same source layout, input asset links, frozen config and audit IDs. Use `results/enhance_bandwidth_fairness.py` for the first three stages and `results/enhance_scaling.py` for scaling. `PRIOR` in those scripts points to the original frozen raw results used for the h0 replay assertions.

The projection method receives only its reference arrays, training-fitted preprocessing, fixed projection and online query. Exact truth and analytic variances are computed separately for evaluation; they are not inputs to the online estimator.

## Data and split records

`results/inputs/` contains original data hashes, metadata and train/reference/validation/test row IDs. `audit_locked_ids.json` identifies the previously unused-at-freeze audit rows. These audits were already examined before this task and are not newly unseen or external datasets.

`results/scaling_ids.json` locks the nested GIST references and fixed 100-query audit prefix. Reference IDs are unique, contain the original 1800 in their existing order, and exclude all train/validation/test/audit rows. Preprocessing, h0 and R0 remain fixed. Raw feature matrices are not duplicated in this delivery; their existing paths/download logic are in `source/run.py` and the prior reproducibility bundle.

## Rebuild analysis locally

Python requires NumPy, pandas and matplotlib. The existing Codex Python runtime was used. From this directory:

```sh
python scripts/audit_existing_mechanism.py
python scripts/summarize_enhance.py
python scripts/verify_enhancement.py
python scripts/build_reports.py
```

The first and third scripts also refer to the neighboring frozen `sprint_final` directory and the baseline paths in `baseline/MANIFEST.json`. Copying this delivery to a different machine requires adjusting those locations or restoring the original sibling directories. All statistical summaries can be regenerated directly from this delivery's `results/` folder.

## Raw-file interpretation

- Bandwidth: `*_queries.csv` contains 20,000 rows per query set; `*_variance.csv` has the matched-H analytic Single/Multi variances, normalized by truth squared, and H mass recall. Single-call timing fields are provenance only and are not compared against three-repeat timing.
- Formal comparisons: `basic_queries.csv` and `deann_queries.csv` contain the first estimate and component-wise median timing of three calls. `*_timings.csv` retains each call. DEANN's permuted sampler advances between repeats, as in the frozen implementation.
- `actual_samples` is the distinct effective sample count. For DEANN, `kernel_evaluations` includes overlap corrections and can exceed nominal M; `distance_evaluations` includes IVF and neighbor-recalculation work. Projected strata also incurs 64 original-dimensional projection dot products and n low-dimensional scores, logged separately.
- `sorting_and_strata_ns` is a combined stage. `other_ns` in the breakdown is total minus the four displayed stages. The breakdown uses arithmetic means so components add to total; frontier tables use median complete time.
- Validation grids select complete configuration curves. No test-query-specific or test-budget-specific parameter selection is performed. If an anchor equals the original DEANN configuration, its observations are reused rather than duplicated as independent data.
- Bootstrap uses query clusters after averaging sampling seeds, 1,000 resamples and seed20260920. Confidence intervals are descriptive and not adjusted for multiple comparisons. Original manuscript intervals remain unchanged.

## Figures

`analysis/bandwidth_matched_H` shows all bandwidths and their paired intervals. `analysis/deann_stress_latency` shows the original reference-size test comparison including all selected DEANN curves. Colors indicate methods; circle/triangle/square/diamond/pentagon mark M32/64/128/256/512, and a star is exact sum. Lines join observations only.

`analysis/scaling_M128` compares error and median time as n grows. DEANN anchor parameters can differ across n because each is selected on its corresponding validation problem using the fixed rule. `analysis/scaling_breakdown` shows mean complete-query components at M128. PNG and vector PDF versions are included for author inspection; none has been inserted into the frozen manuscript.

## Verification

`qa/VERIFICATION.json` records successful budget/count checks, disjoint split checks, executed source hashes, validation-selection reconstruction, timing additivity, n1800 replay, 175 transferred remote-file hashes, and 277 unchanged baseline-file hashes. `qa/TRANSFER_HASHES.json` records the three downloaded archive checksums. An initial extraction attempted before a transfer finished produced a truncated-input error; extraction was repeated only after transfer completion and the complete archive's SHA256 matched the server. No partial extraction was used in final analysis.

The only unresolved intake issue is the literal `(1).pdf` filename. The exact inspected latest-delivery PDF and TeX are copied under `baseline/`; their source locations and hashes identify the assessed version without relying on a filename alone.
