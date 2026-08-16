from pathlib import Path
import databento as db
import polars as pl
from features import OrderBookNormalizer

RAW_INPUT_PATH = Path("data/raw/xnas-itch-20210128.mbp-10.dbn.zst")
PROCESSED_OUTPUT_PATH = Path("data/processed/GME_features_20210128.parquet")

def run_pipeline() -> None:
    if not RAW_INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing file: {RAW_INPUT_PATH}")

    print("[INFO] Loading DBN binary...")
    dbn_store = db.DBNStore.from_file(RAW_INPUT_PATH)
    df_raw = pl.from_pandas(dbn_store.to_df().reset_index())

    print("[INFO] Computing feature matrices...")
    normalizer = OrderBookNormalizer(num_levels=10)
    df_features = normalizer.process_dbn_dataframe(df_raw)

    selected_features = normalizer.feature_columns
    final_columns = ["ts_event", "mid_price", "target_log_return"] + selected_features
    df_final = df_features.select(final_columns)

    print(f"[INFO] Serialization targeting: {PROCESSED_OUTPUT_PATH}")
    PROCESSED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_final.write_parquet(PROCESSED_OUTPUT_PATH, compression="zstd", compression_level=3)

if __name__ == "__main__":
    run_pipeline()