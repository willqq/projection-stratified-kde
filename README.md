# Projection-Guided Remainder Stratification for KDE

Code and reproducibility materials for **Projection-Guided Remainder
Stratification for High-Dimensional Kernel Density Evaluation**, by Wenxin Zheng
and Zhipeng Qiu. This is a manuscript prepared for ICASSP 2027; no acceptance
status is claimed.

A fixed 64-dimensional Gaussian projection provides auxiliary distance scores.
The first `floor(M/4)` references form an exactly evaluated subset H. The rest
form eight equal-population score strata, with proportional sampling. Every
kernel value uses the preprocessed original features and the Gaussian bandwidth
remains fixed. The estimator covers the full reference population.

At M=128 the paper reports 17.7–24.5% lower mean relative error than a single
uniform remainder with identical H and budget. Previously unused-at-freeze
same-source audit queries support the increment. Complete-query advantages are
limited to the measured operating regions. Narrow bandwidths weaken the gain.
The historical norm-angle construction is retained as a diagnostic.

## Quick start

Use Python 3.9–3.12 in a new virtual environment:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python -m pip install --no-deps -e .
python scripts/verify_core.py
python scripts/reproduce_tables.py
python scripts/plot_latency.py
```

The last two commands recompute Tables 1/2 and plot the complete-query
error/latency observations from the saved per-query records. They do not
remeasure runtime. `outputs/paper/TABLE_CHECK.json` checks the printed table
values against the frozen manuscript. All five budgets are retained.

## Run the frozen projection method on the real data

```sh
python scripts/fetch_data.py
python scripts/check_data.py --data-root data/raw
python scripts/run_projection.py --data-root data/raw --dataset isolet \
  --phase test --out outputs/isolet_test.csv
```

Dataset names: `isolet`, `cifar10`, `cifar10_gist512`, `amazon`.
Use `--phase audit` for the first three. Amazon has no unused audit rows.
The default is all 200 queries, five budgets and five sampling seeds. A smaller
`--queries` prefix is a smoke check only. The runner refuses to overwrite outputs.

This portable runner checks statistical replication of Projected/Multi and
matched Single. It removes the unused legacy MECH initialization. Its local
runtime is **not** the manuscript latency, and it does not replace the official
DEANN experiment. Historical complete-query timing replay requires the Linux
environment and frozen experiment source described in `docs/REPRODUCING.md`.

## Layout

```text
src/projected_kde/    portable API and byte-identical frozen numerical modules
scripts/             data checks, core equivalence, statistical replay, tables
configs/             frozen method, dataset rows, audit rows and DEANN anchors
data/                original-file checksums and acquisition instructions
experiments/         original experiment and analysis scripts
third_party/deann/   official DEANN source with its MIT license
results/             compressed paper inputs and published supplemental summaries
reproduction/        original source/assets archive, Git bundle and upgrade source
provenance/          imported-file hashes, source identities and verification
docs/                result-to-code map and historical replay instructions
paper/tables/        frozen printed tables used only for output verification
```

Complete per-query supplementary records, including timing repetitions and
partition caches, are separate release attachments. Their names and checksums
are recorded in `provenance/RELEASE_ASSETS.json`. The repository has not yet
assigned a GitHub release URL; do not interpret the manifest as a live download
link. The included inputs suffice to rebuild Tables 1/2 and the latency curves.

## Reproducibility boundaries

`provenance/VERIFICATION.md` states what was actually executed when preparing
this package. A statistical replay on another machine does not reproduce an
absolute CPU latency. The full historical DEANN/native-extension environment
has not yet been rebuilt on a fresh Linux host for this release.

The prior exposed test queries and same-source audits are distinguished in all
records. No new experiments, parameter selection or manuscript result changes
are part of packaging. The final E/T DEANN validation choices are preserved;
the original weaker DEANN configuration is only an additional named curve.

## License and citation

Author-code licensing is awaiting the authors' selection; see `LICENSE_STATUS.md`.
The vendored DEANN code retains its own MIT license. Raw benchmark data are not
bundled and retain their original terms. Cite the manuscript title and authors;
do not invent a proceedings DOI before one is assigned.
