from pathlib import Path
import polars as pl
import pytest

from src.features import OrderBookNormalizer

# SPY, XNAS.ITCH MBP-10, 2021-01-28 (data/raw/metadata.json), built by the current
# src/features.py with prices in dollars.
PROCESSED_DATA_PATH = Path("data/processed/SPY_features_20210128.parquet")

@pytest.fixture(scope="module")
def feature_matrix():
    if not PROCESSED_DATA_PATH.exists():
        pytest.skip(f"Processed file missing at {PROCESSED_DATA_PATH}. Run pipeline first.")

    # The first 10k rows, 0.22 percent of the file and all from the open. Enough to catch a
    # broken transform, not enough to prove an invariant holds all session.
    return pl.read_parquet(PROCESSED_DATA_PATH).head(10000)

def test_no_null_values_leakage(feature_matrix):
    # null_count does not count NaN, which is a valid float to Polars.
    total_nulls = sum(feature_matrix.null_count().row(0))
    assert total_nulls == 0, f"Matrix contains {total_nulls} nulls."

def test_temporal_discretization_bounds(feature_matrix):
    min_delta = feature_matrix.select(pl.col("delta_t").min()).item()
    assert min_delta >= 0.0, f"Negative time jump detected: {min_delta}"

def test_spread_is_strictly_positive(feature_matrix):
    # bid == ask is a locked book and bid > ask a crossed one. Neither should survive the
    # MBP-10 parse, so either is a pipeline bug rather than a market event.
    min_spread = feature_matrix.select(pl.col("spread_level_0").min()).item()
    assert min_spread > 0.0, f"Crossed book or zero spread detected: {min_spread}"

def test_prices_are_in_dollars(feature_matrix):
    # SPY traded between 370 and 382 dollars that day. A mid near 3.8e-7 means the 1e-9
    # fixed-point scale was applied twice, once by DBNStore.to_df() and once more here.
    mid_min = feature_matrix.select(pl.col("mid_price").min()).item()
    mid_max = feature_matrix.select(pl.col("mid_price").max()).item()
    assert 300.0 < mid_min <= mid_max < 450.0, f"mid_price outside SPY's range: {mid_min}, {mid_max}"

def test_spread_is_at_least_one_tick(feature_matrix):
    # Rule 612 of Reg NMS bars sub-penny quotes above one dollar, so a quoted spread of
    # less than 0.01 dollars is a scale error.
    min_spread = feature_matrix.select(pl.col("spread_level_0").min()).item()
    assert min_spread >= 0.01 - 1e-9, f"Spread below one cent: {min_spread}"

def test_feature_definitions_without_rescaling():
    # Dollar prices pass through unchanged: mid, spread, distances and micro-price on a
    # hand-checked two-row book, with the target over a horizon cut to fit.
    n = 2
    row = {"ts_event": [0, 1_000_000_000]}
    for i in range(10):
        row[f"ask_px_{i:02d}"] = [100.02 + 0.01 * i] * n
        row[f"bid_px_{i:02d}"] = [100.00 - 0.01 * i] * n
        row[f"ask_sz_{i:02d}"] = [3] * n
        row[f"bid_sz_{i:02d}"] = [1] * n
    df = pl.DataFrame(row).with_columns(pl.col("ts_event").cast(pl.Datetime("ns")))
    df = df.with_columns(pl.col(c).cast(pl.UInt32) for c in df.columns if "_sz_" in c)

    out = OrderBookNormalizer(num_levels=10).process_dbn_dataframe(
        pl.concat([df] * 51)  # 102 rows, so the horizon-100 shift leaves 2
    )
    assert out.height == 2
    first = out.row(0, named=True)
    assert first["mid_price"] == pytest.approx(100.01)
    assert first["spread_level_0"] == pytest.approx(0.02)
    assert first["ask_dist_00"] == pytest.approx(0.01)
    assert first["bid_dist_09"] == pytest.approx(-0.10)
    assert first["obi_level_0"] == pytest.approx(-0.5)
    # (bid * ask_sz + ask * bid_sz) / (bid_sz + ask_sz), weighted towards the thinner side
    assert first["micro_price"] == pytest.approx((100.00 * 3 + 100.02 * 1) / 4)
    assert first["target_log_return"] == pytest.approx(0.0, abs=1e-15)
    assert first["delta_t"] == 0.0

def test_spatial_distance_geometries(feature_matrix):
    # distances are signed relative to mid, so asks come out >= 0 and bids <= 0
    ask_dist_min = feature_matrix.select(pl.col("ask_dist_00").min()).item()
    assert ask_dist_min >= 0.0, f"Ask distance invariant violated: {ask_dist_min}"

    bid_dist_max = feature_matrix.select(pl.col("bid_dist_00").max()).item()
    assert bid_dist_max <= 0.0, f"Bid distance invariant violated: {bid_dist_max}"

def test_order_book_monotonicity(feature_matrix):
    # prices get worse with depth, so in mid-relative terms ask_dist grows with the level
    # index and bid_dist gets more negative. Non-strict: equal adjacent levels are legal.
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
    # 3 bookkeeping + 4 derived + 10 levels * 4 = 47 columns in the parquet. src/dataset.py
    # then drops 4 of them, which is where the model's 43 inputs come from.
    expected_cols = 47
    actual_cols = len(feature_matrix.columns)
    assert actual_cols == expected_cols, f"Tensor shape mismatch. Expected {expected_cols}, got {actual_cols}"