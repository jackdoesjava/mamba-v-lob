"""Turn a raw Databento MBP-10 file into the processed feature parquet.

See docs/08-data.md for the feature definitions and invariants.
"""

from pathlib import Path

import databento as db
import polars as pl

from src.features import OrderBookNormalizer
from src.utils.config import load_config

RAW_INPUT_PATH = Path("data/raw/xnas-itch-20210128.mbp-10.dbn.zst")


def run_pipeline() -> None:
    if not RAW_INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing file: {RAW_INPUT_PATH}")

    output_path = Path(load_config()["data"]["processed_file"])

    print("[INFO] Loading DBN binary...")
    dbn_store = db.DBNStore.from_file(RAW_INPUT_PATH)
    df_raw = pl.from_pandas(dbn_store.to_df().reset_index())

    print("[INFO] Computing feature matrices...")
    normalizer = OrderBookNormalizer(num_levels=10)
    df_features = normalizer.process_dbn_dataframe(df_raw)

    # mid_price and ts_event are kept for inspection; src/dataset.py drops them before
    # they can reach the model.
    final_columns = ["ts_event", "mid_price", "target_log_return"] + normalizer.feature_columns
    df_final = df_features.select(final_columns)

    print(f"[INFO] Serialization targeting: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.write_parquet(output_path, compression="zstd", compression_level=3)


if __name__ == "__main__":
    run_pipeline()
