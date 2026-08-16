from pathlib import Path
import polars as pl
import pytest

# Target the generated file
PROCESSED_DATA_PATH = Path("data/processed/GME_features_20210128.parquet")

@pytest.fixture(scope="module")
def feature_matrix():
    """Loads a 10,000-row chunk of the processed data for rapid unit testing."""
    if not PROCESSED_DATA_PATH.exists():
        pytest.skip(f"Processed file missing at {PROCESSED_DATA_PATH}. Run pipeline first.")
    
    # Read a sample to keep tests sub-second
    return pl.read_parquet(PROCESSED_DATA_PATH).head(10000)

def test_no_null_values_leakage(feature_matrix):
    # Sum all nulls across all columns. Must be exactly 0.
    total_nulls = sum(feature_matrix.null_count().row(0))
    assert total_nulls == 0, f"Matrix contains {total_nulls} NaN/Null values."

def test_temporal_discretization_bounds(feature_matrix):
    # Time must only move forward (delta_t >= 0)
    min_delta = feature_matrix.select(pl.col("delta_t").min()).item()
    assert min_delta >= 0.0, f"Negative time jump detected: {min_delta}"

def test_spread_is_strictly_positive(feature_matrix):
    # The limit order book cannot be crossed (ask must be > bid)
    min_spread = feature_matrix.select(pl.col("spread_level_0").min()).item()
    assert min_spread > 0.0, f"Crossed book or zero spread detected: {min_spread}"

def test_spatial_distance_geometries(feature_matrix):
    # Asks must sit above the mid-price (distance >= 0)
    ask_dist_min = feature_matrix.select(pl.col("ask_dist_00").min()).item()
    assert ask_dist_min >= 0.0, f"Ask distance invariant violated: {ask_dist_min}"
    
    # Bids must sit below the mid-price (distance <= 0)
    bid_dist_max = feature_matrix.select(pl.col("bid_dist_00").max()).item()
    assert bid_dist_max <= 0.0, f"Bid distance invariant violated: {bid_dist_max}"

def test_order_book_monotonicity(feature_matrix):
    # LOB Physics: The deeper you go into the book, the worse the prices must get.
    # Asks must strictly increase (or stay flat). Bids must strictly decrease (or stay flat).
    # Since we converted to distances from mid-price:
    # ask_dist_01 must be >= ask_dist_00
    ask_diff = feature_matrix.select((pl.col("ask_dist_01") - pl.col("ask_dist_00")).min()).item()
    assert ask_diff >= 0.0, f"Ask book is crossed with itself: {ask_diff}"
    
    # bid_dist_01 must be <= bid_dist_00 (more negative)
    bid_diff = feature_matrix.select((pl.col("bid_dist_00") - pl.col("bid_dist_01")).min()).item()
    assert bid_diff >= 0.0, f"Bid book is crossed with itself: {bid_diff}"

def test_obi_bounds(feature_matrix):
    # Order Book Imbalance (OBI) is a normalized ratio. 
    # It must strictly sit between -1.0 (pure ask pressure) and +1.0 (pure bid pressure).
    obi_min = feature_matrix.select(pl.col("obi_level_0").min()).item()
    obi_max = feature_matrix.select(pl.col("obi_level_0").max()).item()
    assert obi_min >= -1.0 and obi_max <= 1.0, f"OBI broke bounds: min={obi_min}, max={obi_max}"

def test_feature_tensor_dimensionality(feature_matrix):
    # Mamba requires a statically sized input dimension.
    # 3 metadata cols + 4 alpha cols + (10 levels * 4 features) = 47 columns.
    expected_cols = 47
    actual_cols = len(feature_matrix.columns)
    assert actual_cols == expected_cols, f"Tensor shape mismatch. Expected {expected_cols}, got {actual_cols}"