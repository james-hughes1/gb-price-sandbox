import pandas as pd

from gb_price_sandbox.data.cache import cached_parquet


def test_cache_is_idempotent(tmp_path):
    calls = []

    def fetch():
        calls.append(1)
        return pd.DataFrame({"price": [1.0, 2.0]})

    path = tmp_path / "sub" / "prices.parquet"
    first = cached_parquet(path, fetch)
    second = cached_parquet(path, fetch)

    assert len(calls) == 1
    pd.testing.assert_frame_equal(first, second)


def test_refresh_refetches(tmp_path):
    calls = []

    def fetch():
        calls.append(1)
        return pd.DataFrame({"price": [float(len(calls))]})

    path = tmp_path / "prices.parquet"
    cached_parquet(path, fetch)
    out = cached_parquet(path, fetch, refresh=True)

    assert len(calls) == 2
    assert out["price"].iloc[0] == 2.0
