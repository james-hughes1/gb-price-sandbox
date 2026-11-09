import pandas as pd

from gb_price_sandbox.data.cache import cached_monthly, cached_parquet


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


def test_cached_monthly_skips_writing_recent_months(tmp_path):
    calls = []

    def fetch_range(a, b):
        calls.append((a, b))
        idx = pd.date_range(a, b, freq="1D", inclusive="left")
        return pd.DataFrame({"x": 1.0}, index=idx)

    utc = "UTC"
    start, end = pd.Timestamp("2024-01-15", tz=utc), pd.Timestamp("2024-03-10", tz=utc)
    now = pd.Timestamp("2024-03-20", tz=utc)  # March is not complete yet

    out = cached_monthly(tmp_path, start, end, fetch_range, now=now)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["2024-01.parquet", "2024-02.parquet"]
    assert out.index.min() == start and out.index.max() == pd.Timestamp("2024-03-09", tz=utc)

    calls.clear()
    cached_monthly(tmp_path, start, end, fetch_range, now=now)
    assert [a.month for a, _ in calls] == [3]  # only the open month is re-fetched
