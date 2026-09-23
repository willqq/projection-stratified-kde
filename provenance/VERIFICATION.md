# Executed checks for this package

A fresh Python 3.12 virtual environment was created on macOS arm64. The pinned analysis dependencies and editable package installed successfully.

- Both manuscript tables were rebuilt from compressed per-query observations. All printed values, including paired bootstrap intervals, match.
- The portable wrapper was compared with the unmodified frozen evaluator class in 200 evaluations: selected IDs, H, log estimates and sample budgets match exactly. Census evaluation also matches the full kernel mean.
- Four real dataset files were downloaded through the commit-pinned mirror. All four SHA-256 values match the frozen experimental manifests.
- Each real dataset was replayed on its first 10 frozen test queries, all five budgets, five sampling seeds and both Single/Multi. All 2,000 observations match saved errors within 2.7e-15; actual sample counts match exactly.
- The latency plot was regenerated from saved complete-query timings. No new timing numbers replace manuscript observations.

The real-data check is a 10-query-per-dataset replication check, not a new full 200-query experiment. The source runner supports all 200 queries. Official DEANN and native posting extensions were not rebuilt or timed on a fresh Linux host. Complete historical source, binary assets and replay instructions are preserved for that route; this remaining boundary is not hidden by the portable checks.

Machine-readable records are next to this file. The original method, frozen results and source directories were not modified.
