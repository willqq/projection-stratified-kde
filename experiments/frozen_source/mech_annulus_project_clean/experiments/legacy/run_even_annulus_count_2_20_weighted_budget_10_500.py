from __future__ import annotations

from pathlib import Path

import run_even_annulus_count_2_20_weighted_budget_sensitivity as experiment


experiment.OUT_DIR = Path(
    "mech_annulus_count_even_2_20_weighted_budget_sensitivity_2x_anchor_budget_10_500"
)
experiment.TOTAL_SAMPLE_BUDGETS = list(range(10, 501, 10))


if __name__ == "__main__":
    experiment.main()
