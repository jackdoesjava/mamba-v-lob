from pathlib import Path
import polars as pl
import pytest

PROCESSED_DATA_PATH = Path("data/processed/GME_features_20210128.parquet")

@pytest.fixture(scope="module")
def feature_matrix():
    if not PROCESSED_DATA_PATH.exists():
        pytest.skip(f"Processed file missing at {PROCESSED_DATA_PATH}. Run pipeline first.")

    # 10k rows is plenty to trip any invariant and keeps the suite sub-second
    return pl.read_parquet(PROCESSED_DATA_PATH).head(10000)

def test_no_null_values_leakage(feature_matrix):
    total_nulls = sum(feature_matrix.null_count().row(0))
    assert total_nulls == 0, f"Matrix contains {total_nulls} NaN/Null values."

def test_temporal_discretization_bounds(feature_matrix):
    min_delta = feature_matrix.select(pl.col("delta_t").min()).item()
    assert min_delta >= 0.0, f"Negative time jump detected: {min_delta}"

def test_spread_is_strictly_positive(feature_matrix):
    # a zero or negative spread means a crossed book, which is a parsing bug rather than a market event
    min_spread = feature_matrix.select(pl.col("spread_level_0").min()).item()
    assert min_spread > 0.0, f"Crossed book or zero spread detected: {min_spread}"

def test_spatial_distance_geometries(feature_matrix):
    # distances are signed relative to mid, so asks come out >= 0 and bids <= 0
    ask_dist_min = feature_matrix.select(pl.col("ask_dist_00").min()).item()
    assert ask_dist_min >= 0.0, f"Ask distance invariant violated: {ask_dist_min}"

    bid_dist_max = feature_matrix.select(pl.col("bid_dist_00").max()).item()
    assert bid_dist_max <= 0.0, f"Bid distance invariant violated: {bid_dist_max}"

def test_order_book_monotonicity(feature_matrix):
    # prices get worse with depth, so in mid-relative terms ask_dist grows with
    # the level index and bid_dist gets more negative
    ask_diff = feature_matrix.select((pl.col("ask_dist_01") - pl.col("ask_dist_00")).min()).item()
    assert ask_diff >= 0.0, f"Ask book is crossed with itself: {ask_diff}"

    bid_diff = feature_matrix.select((pl.col("bid_dist_00") - pl.col("bid_dist_01")).min()).item()
    assert bid_diff >= 0.0, f"Bid book is crossed with itself: {bid_diff}"

def test_obi_bounds(feature_matrix):
    # OBI is a normalised ratio: -1 is all ask pressure, +1 all bid.
    obi_min = feature_matrix.select(pl.col("obi_level_0").min()).item()
    obi_max = feature_matrix.select(pl.col("obi_level_0").max()).item()
    assert obi_min >= -1.0 and obi_max <= 1.0, f"OBI broke bounds: min={obi_min}, max={obi_max}"

def test_feature_tensor_dimensionality(feature_matrix):
    # 3 metadata + 4 alpha + 10 levels * 4 = 47, and the model needs that fixed
    expected_cols = 47
    actual_cols = len(feature_matrix.columns)
    assert actual_cols == expected_cols, f"Tensor shape mismatch. Expected {expected_cols}, got {actual_cols}"