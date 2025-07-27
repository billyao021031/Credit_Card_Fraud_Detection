from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW  = PROJECT_ROOT / "data" / "creditcard.csv"
DEFAULT_OUT  = PROJECT_ROOT / "data" / "processed" / "creditcard_processed.parquet"
SECS_IN_DAY  = 86_400


def _cyclic_time_features(time_col: pd.Series) -> pd.DataFrame:
    """Return sin/cos of the daily phase derived from `Time` in *seconds*."""
    phase = 2 * np.pi * (time_col % SECS_IN_DAY) / SECS_IN_DAY
    return pd.DataFrame({
        "sin_time": np.sin(phase).astype(np.float32),
        "cos_time": np.cos(phase).astype(np.float32),
    })


def make_dataset(raw_path: Path, out_path: Path) -> None:
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw CSV not found → {raw_path}")

    df = pd.read_csv(raw_path)
    df["Amount_log"] = np.log1p(df["Amount"]).astype(np.float32)

    # z‑score the Time column
    time_mean = df["Time"].mean()
    time_std  = df["Time"].std()
    df["Time_z"] = ((df["Time"] - time_mean) / time_std).astype(np.float32)

    df = pd.concat([df, _cyclic_time_features(df["Time"])], axis=1)

    # ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    rel = out_path.relative_to(PROJECT_ROOT)
    print(f"Processed dataset saved → {rel}  (rows: {len(df):,})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Preprocess ULB credit-card dataset")
    parser.add_argument("--input", default=str(DEFAULT_RAW), help="raw CSV path")
    parser.add_argument("--output", default=str(DEFAULT_OUT), help="output parquet path")
    args = parser.parse_args()

    make_dataset(Path(args.input), Path(args.output))
