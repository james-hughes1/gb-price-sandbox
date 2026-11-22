"""Command line: ``python -m gb_price_sandbox.pretrain`` saves the Battery page's forecasts."""

from __future__ import annotations

import warnings


def main() -> None:
    """Pretrain the Battery optimiser's models over the default holdout."""
    from gb_price_sandbox import app_data
    from gb_price_sandbox.config import get_settings
    from gb_price_sandbox.pretrained import pretrain

    warnings.filterwarnings("ignore")
    out = pretrain(
        app_data.load_study_data(),
        app_data.DEFAULT_TEST_START,
        app_data.default_test_end(),
        get_settings(),
        progress=print,
    )
    print(
        f"Saved {', '.join(out.results)} forecasts for "
        f"{out.test_start:%d %b %Y} to {out.test_end:%d %b %Y}."
    )


if __name__ == "__main__":
    main()
