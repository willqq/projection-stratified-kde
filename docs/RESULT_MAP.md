# Claims and reproduction entry points

| Manuscript evidence | Preserved observations | Code |
|---|---|---|
| Table 1 and final latency figure | `results/paper_inputs/deann_formal/` | `scripts/reproduce_tables.py`, `scripts/plot_latency.py`; original `enhance_bandwidth_fairness.py` |
| Table 2A, fixed-population variance | `results/paper_inputs/fixed_c/` | `scripts/reproduce_tables.py`; original `sprint_fixed_c_final.py` |
| Table 2B, same H and budget | `results/paper_inputs/sprint/` | `scripts/reproduce_tables.py`; original `sprint_formal.py` |
| Projection seed check | experiment-records attachment, `structural_revision_20260919/remote/` | `experiments/analysis/structural_revision_20260919/` |
| Bandwidth and stronger DEANN validation | experiment-records attachment, `enhancement_assessment_20260920/results/` | `experiments/drivers/enhance_bandwidth_fairness.py` |
| Nested real GIST reference-size scaling | same attachment, scaling/scaling_validation plus `configs/scaling_ids.json` | `experiments/drivers/enhance_scaling.py` |
| Fixed-union and ordering interventions | same attachment, `mechanism_assessment_20260920/results/` | `experiments/drivers/mechanism.py` and its analysis scripts |
| ANN-H controls and equivalent construction | upgrade-raw-records attachment | `experiments/frozen_source/mech_annulus_project_clean/revision/upgrade_*.py` |

The original query CSVs contain sampling-seed identifiers and raw row IDs.
Original timing repetitions remain in the attachments. The table script averages
sampling seeds inside each query before paired query bootstrap. It preserves the
original bootstrap stream used in the manuscript.

Saved summaries in `results/published/` provide convenient inspection. They are
not substituted for query records when rebuilding the main tables. The current
paper's Table 1 uses the stronger DEANN-E experiment; using only the September
16 sprint table would reproduce an earlier paper revision.
