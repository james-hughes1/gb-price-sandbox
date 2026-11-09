from pathlib import Path

from gb_price_sandbox.config import Settings


def test_defaults_when_env_empty():
    s = Settings.from_env({})
    assert s == Settings()


def test_env_overrides():
    s = Settings.from_env({"GBPS_DATA_DIR": "/tmp/x", "GBPS_HTTP_TIMEOUT": "5"})
    assert s.data_dir == Path("/tmp/x")
    assert s.http_timeout == 5.0
