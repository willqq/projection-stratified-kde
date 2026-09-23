from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.stats import binom

import run_r25_lk_paper_angle_grid_sensitivity as base
import run_r25_l1_25_k10_100_recall_boosted_angle_grid_sensitivity as boosted
from run_r25_k10_200_l20_empirical_angle_lookup import (
    layer_max_angle_from_radius,
    paper_layer_bounds,
)


OUT_DIR = Path("r25_l1_25_k10_100_high_recall_angle_grid_sensitivity")
TARGET_SINGLE_TABLE_CDF_AT_MAX_K = 0.995
EXTRA_MARGIN_RATIO_AT_MAX_K = 0.18
MIN_THRESHOLD_RATIO_AT_MAX_K = 0.35


def k_progress(bits: int) -> float:
    if max(boosted.K_VALUES) <= min(boosted.K_VALUES):
        return 0.0
    return float(
        np.clip(
            (bits - min(boosted.K_VALUES))
            / (max(boosted.K_VALUES) - min(boosted.K_VALUES)),
            0.0,
            1.0,
        )
    )


def high_recall_hamming_threshold_from_angle(
    theta: float,
    bits: int,
    tables: int,
    alpha: float,
) -> int:
    bit_flip_probability = float(np.clip(theta / math.pi, 0.0, 1.0))
    strict_target = 1.0 - (1.0 - alpha) ** (1.0 / tables)
    progress = k_progress(bits)
    target = max(
        strict_target,
        strict_target
        + progress * (TARGET_SINGLE_TABLE_CDF_AT_MAX_K - strict_target),
    )
    threshold = int(binom.ppf(target, bits, bit_flip_probability))
    threshold += int(math.ceil(EXTRA_MARGIN_RATIO_AT_MAX_K * progress * bits))

    min_ratio = MIN_THRESHOLD_RATIO_AT_MAX_K * progress
    threshold = max(threshold, int(math.ceil(min_ratio * bits)))
    return min(max(threshold, 0), bits)


def main() -> None:
    boosted.OUT_DIR = OUT_DIR
    boosted.TARGET_SINGLE_TABLE_CDF_AT_MAX_K = TARGET_SINGLE_TABLE_CDF_AT_MAX_K
    boosted.EXTRA_MARGIN_RATIO_AT_MAX_K = EXTRA_MARGIN_RATIO_AT_MAX_K
    boosted.MIN_THRESHOLD_RATIO_AT_MAX_K = MIN_THRESHOLD_RATIO_AT_MAX_K
    boosted.recall_boosted_hamming_threshold_from_angle = (
        high_recall_hamming_threshold_from_angle
    )
    boosted.configure_base()

    base.paper_layer_bounds = paper_layer_bounds
    base.layer_angle_from_radius = layer_max_angle_from_radius
    base.main()


if __name__ == "__main__":
    main()
