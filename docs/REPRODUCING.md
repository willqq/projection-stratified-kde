# Three reproducibility tasks

## Recompute the paper's printed evidence

Follow the README quick start. This needs no raw feature data or native DEANN
extension. It uses the saved, complete per-query inputs for Tables 1/2. The plot
contains all measured methods and budgets, including unfavorable outcomes.
Rendered plot styling may differ; the underlying measurements are unchanged.

## Rerun the final estimator

`scripts/run_projection.py` uses the exact frozen projection and allocation
modules, original data hashes, split IDs, h and sampling seeds. `verify_core.py`
compares against the original evaluator class, including the selected IDs and
log estimates. It avoids loading the legacy neural hash model when only the
projection path is requested. No new score or estimator is introduced.

This is a new-host statistical replication. The production timing runner and
its counts remain available separately. Do not replace published times with
wrapper times or compare a wrapper against a separately timed DEANN run.

## Restore the complete historical production experiment

The measured environment used Linux x86-64, CPython 3.8.10, eight CPU threads,
NumPy 1.24.2, SciPy 1.10.1, Torch 2.0.0+cu118 and FAISS 1.7.4. CUDA was disabled
for formal timings. See `reproduction/base/requirements.runtime.txt` and the
environment record. These differ from the portable analysis environment.

Start in a fresh directory, preserving all saved results:

```sh
mkdir replay
git clone --branch icassp-sprint-20260916 \
  reproduction/base/frozen_branch.bundle replay/worktree
tar -xzf reproduction/base/code_assets.tar.gz -C replay
git -C replay/worktree rev-parse HEAD
git -C replay/worktree diff --exit-code
```

The base HEAD must be `59d86dbfaaeebb7bdaace151afcec047f3e1a360`.
Copy or link the checked raw data beneath `replay/worktree/data/`. Follow
`docs/historical/sprint_final.md` for the preserved native extension and MKL
environment, exact commands, model assets and formal freeze checks. Its example
server paths must be adjusted to the new location. The freeze includes a native
binary hash: a rebuilt extension requires a separately documented replication
manifest. Do not falsify an old hash or disable checks to claim exact replay.

Later experiments are preserved under `experiments/` and `reproduction/upgrade/`.
Their historical instructions are in `docs/historical/`. They include hard-coded
original paths for prior-results assertions; relocate these in a separately
recorded replication copy. Frozen originals in this repository are unchanged.

The final upgrade experiment source commit is
`f872802ae4cca95ec69376d548c8389627de8761`. Its final source archive is
`757d1801748d5b19ca4940d7a99d89d0b9f86ed8`; later differences are analysis and
reporting. `reproduction/upgrade/SOURCE_STATE.json` explains those identities.

This package preserves the complete source/assets needed for this historical
route, with a buildable official DEANN source. A new Linux rebuild and all-method
end-to-end rerun remain an explicit verification task; they have not been
replaced with the portable statistical checks.
