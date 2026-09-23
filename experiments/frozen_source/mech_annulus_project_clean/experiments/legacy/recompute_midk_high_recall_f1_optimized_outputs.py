from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import run_r25_lk_paper_angle_grid_sensitivity as base
import run_r25_l1_25_k10_100_recall_boosted_angle_grid_sensitivity as boosted


SOURCE_DIR = Path("r25_l1_25_k10_100_midk_high_recall_angle_grid_sensitivity")
OUT_DIR = Path("r25_l1_25_k10_100_midk_high_recall_f1_optimized")


def main() -> None:
    source_csv = SOURCE_DIR / "r25_l1_25_k10_100_recall_boosted_angle_grid_all.csv"
    config_path = SOURCE_DIR / "config.json"

    results = pd.read_csv(source_csv)
    metadata = json.loads(config_path.read_text(encoding="utf-8")).get("metadata", [])

    boosted.OUT_DIR = OUT_DIR
    boosted.TARGET_SINGLE_TABLE_CDF_AT_MAX_K = 0.995
    boosted.EXTRA_MARGIN_RATIO_AT_MAX_K = 0.18
    boosted.MIN_THRESHOLD_RATIO_AT_MAX_K = 0.35
    boosted.FULL_BOOST_K = 50
    boosted.configure_base()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    boosted.write_outputs(results, metadata)

    # The writer intentionally preserves the original all-grid filename. Add a
    # short marker config so this derived directory is unambiguous.
    config = json.loads((OUT_DIR / "config.json").read_text(encoding="utf-8"))
    config["derived_from"] = str(SOURCE_DIR)
    config["objective"] = "f1_optimized"
    (OUT_DIR / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
