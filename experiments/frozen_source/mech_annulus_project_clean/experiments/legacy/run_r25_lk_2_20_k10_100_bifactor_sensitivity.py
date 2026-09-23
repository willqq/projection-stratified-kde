from __future__ import annotations

from pathlib import Path

import pandas as pd

import run_r25_lk_paper_angle_grid_sensitivity as grid


grid.OUT_DIR = Path("r25_lk_2_20_k10_100_bifactor_sensitivity")
grid.L_VALUES = list(range(2, 21))
grid.K_VALUES = list(range(10, 101, 10))
grid.MAX_L = max(grid.L_VALUES)
grid.MAX_K = max(grid.K_VALUES)


def load_local_rows(_: str, result_path: Path) -> pd.DataFrame:
    if result_path.exists():
        return pd.read_csv(result_path)
    return pd.DataFrame()


grid.load_seed_rows = load_local_rows


if __name__ == "__main__":
    grid.main()
