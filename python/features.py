from pathlib import Path
from typing import List
import polars as pl

class OrderBookNormalizer:
    def __init__(self, num_levels: int = 10):
        self.num_levels = num_levels

    def process_dbn_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        PRICE_SCALE = 1e9

        # Nanosecond timestamps to seconds for explicit delta_t execution bounds
        df = df.with_columns([
            (pl.col("ts_event").cast(pl.Int64) / 1e9).alias("timestamp_sec")
        ]).with_columns([
            (pl.col("timestamp_sec") - pl.col("timestamp_sec").shift(1))
            .fill_null(0.0)
            .alias("delta_t")
        ])

        price_cols = [f"ask_px_{i:02d}" for i in range(self.num_levels)] + \
                     [f"bid_px_{i:02d}" for i in range(self.num_levels)]
        
        df = df.with_columns([
            (pl.col(col) / PRICE_SCALE).alias(col) for col in price_cols
        ])

        df = df.with_columns([
            ((pl.col("ask_px_00") + pl.col("bid_px_00")) / 2.0).alias("mid_price"),
            (pl.col("ask_px_00") - pl.col("bid_px_00")).alias("spread_level_0")
        ])

        # Enforce spatial stationarity relative to instantaneous mid-price
        stationary_exprs = []
        for i in range(self.num_levels):
            stationary_exprs.append((pl.col(f"ask_px_{i:02d}") - pl.col("mid_price")).alias(f"ask_dist_{i:02d}"))
            stationary_exprs.append((pl.col(f"bid_px_{i:02d}") - pl.col("mid_price")).alias(f"bid_dist_{i:02d}"))
        
        # Log scaling stabilizes volumetric variance and prevents gradient explosion in long sequences
        for i in range(self.num_levels):
            stationary_exprs.append((pl.col(f"ask_sz_{i:02d}") + 1.0).log().alias(f"ask_log_sz_{i:02d}"))
            stationary_exprs.append((pl.col(f"bid_sz_{i:02d}") + 1.0).log().alias(f"bid_log_sz_{i:02d}"))

        df = df.with_columns(stationary_exprs)

        # Cast to float to prevent unsigned integer underflow when ask_sz > bid_sz
        bid_sz_f = pl.col("bid_sz_00").cast(pl.Float64)
        ask_sz_f = pl.col("ask_sz_00").cast(pl.Float64)

        df = df.with_columns([
            ((bid_sz_f - ask_sz_f) / 
             (bid_sz_f + ask_sz_f + 1e-8)).alias("obi_level_0"),
            (((pl.col("bid_px_00") * ask_sz_f) + (pl.col("ask_px_00") * bid_sz_f)) / 
             (bid_sz_f + ask_sz_f + 1e-8)).alias("micro_price")
        ])

        # Target label: 100-tick-ahead mid-price log return
        PREDICTION_HORIZON = 100
        df = df.with_columns([
            (pl.col("mid_price").shift(-PREDICTION_HORIZON) / pl.col("mid_price"))
            .log()
            .alias("target_log_return")
        ]).drop_nulls()

        return df

    @property
    def feature_columns(self) -> List[str]:
        features = ["delta_t", "spread_level_0", "obi_level_0", "micro_price"]
        for i in range(self.num_levels):
            features.extend([f"ask_dist_{i:02d}", f"bid_dist_{i:02d}", f"ask_log_sz_{i:02d}", f"bid_log_sz_{i:02d}"])
        return features