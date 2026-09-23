from __future__ import annotations

from pathlib import Path

import run_r25_lk_paper_angle_grid_sensitivity as base
import run_r25_l1_25_k10_100_recall_boosted_angle_grid_sensitivity as boosted
from run_r25_k10_200_l20_empirical_angle_lookup import (
    layer_max_angle_from_radius,
    paper_layer_bounds,
)


OUT_DIR = Path("r25_l1_25_k10_100_midk_high_recall_angle_grid_sensitivity")
TARGET_SINGLE_TABLE_CDF_AT_MAX_K = 0.995
EXTRA_MARGIN_RATIO_AT_MAX_K = 0.18
MIN_THRESHOLD_RATIO_AT_MAX_K = 0.35
FULL_BOOST_K = 50


def main() -> None:
    boosted.OUT_DIR = OUT_DIR
    boosted.TARGET_SINGLE_TABLE_CDF_AT_MAX_K = TARGET_SINGLE_TABLE_CDF_AT_MAX_K
    boosted.EXTRA_MARGIN_RATIO_AT_MAX_K = EXTRA_MARGIN_RATIO_AT_MAX_K
    boosted.MIN_THRESHOLD_RATIO_AT_MAX_K = MIN_THRESHOLD_RATIO_AT_MAX_K
    boosted.FULL_BOOST_K = FULL_BOOST_K
    boosted.configure_base()

    base.paper_layer_bounds = paper_layer_bounds
    base.layer_angle_from_radius = layer_max_angle_from_radius
    base.main()


if __name__ == "__main__":
    main()
