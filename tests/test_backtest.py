import pandas as pd

from stock_system.backtest import BacktestConfig, backtest_strategy, summarize_backtest


def test_backtest_generates_trade_and_return():
    signals = pd.DataFrame(
        [
            {"ticker": "AAA", "date_screened": "2024-01-02", "passes_screen": True},
        ]
    )
    prices = pd.DataFrame(
        [
            {"ticker": "AAA", "date": "2024-01-02", "close": 100},
            {"ticker": "AAA", "date": "2024-07-01", "close": 120},
        ]
    )

    out = backtest_strategy(signals, prices, BacktestConfig(hold_days=180, transaction_cost_bps=0, filing_delay_days=0))
    assert len(out) == 1
    assert out.iloc[0]["buy_price"] == 100
    assert out.iloc[0]["sell_price"] == 120
    assert round(out.iloc[0]["return_pct"], 4) == 20.0


def test_backtest_applies_transaction_costs():
    signals = pd.DataFrame(
        [
            {"ticker": "AAA", "date_screened": "2024-01-02", "passes_screen": True},
        ]
    )
    prices = pd.DataFrame(
        [
            {"ticker": "AAA", "date": "2024-01-02", "close": 100},
            {"ticker": "AAA", "date": "2024-07-01", "close": 110},
        ]
    )

    out = backtest_strategy(signals, prices, BacktestConfig(hold_days=180, transaction_cost_bps=50, filing_delay_days=0))
    # Gross 10%, roundtrip cost 1% => net 9%
    assert round(out.iloc[0]["return_pct"], 4) == 9.0


def test_summarize_backtest_values():
    results = pd.DataFrame(
        [
            {"return_pct": 10.0, "buy_date": "2024-01-02", "sell_date": "2024-07-01"},
            {"return_pct": -5.0, "buy_date": "2024-02-01", "sell_date": "2024-07-30"},
            {"return_pct": 8.0, "buy_date": "2024-03-01", "sell_date": "2024-08-28"},
        ]
    )
    summary = summarize_backtest(results)
    assert summary["num_trades"] == 3
    assert round(summary["avg_return_pct"], 4) == 4.3333
    assert round(summary["win_rate"], 4) == 0.6667


def test_backtest_filing_delay():
    """Buy should be delayed by filing_delay_days after the signal date."""
    signals = pd.DataFrame(
        [{"ticker": "AAA", "date_screened": "2024-01-02", "passes_screen": True}]
    )
    prices = pd.DataFrame(
        [
            {"ticker": "AAA", "date": "2024-01-02", "close": 100},
            {"ticker": "AAA", "date": "2024-02-16", "close": 105},  # 45 days later
            {"ticker": "AAA", "date": "2024-08-14", "close": 130},  # 180 days after buy
        ]
    )
    out = backtest_strategy(
        signals, prices, BacktestConfig(hold_days=180, transaction_cost_bps=0, filing_delay_days=45)
    )
    assert len(out) == 1
    # Should buy on 2024-02-16 (first price on or after signal + 45 days), not 2024-01-02
    assert out.iloc[0]["buy_date"] == "2024-02-16"
    assert out.iloc[0]["buy_price"] == 105


def test_backtest_skips_overlapping_positions():
    """Second signal for same ticker during active holding should be skipped."""
    signals = pd.DataFrame(
        [
            {"ticker": "AAA", "date_screened": "2024-01-02", "passes_screen": True},
            {"ticker": "AAA", "date_screened": "2024-03-01", "passes_screen": True},  # within hold period
        ]
    )
    prices = pd.DataFrame(
        [
            {"ticker": "AAA", "date": "2024-01-02", "close": 100},
            {"ticker": "AAA", "date": "2024-03-01", "close": 110},
            {"ticker": "AAA", "date": "2024-07-01", "close": 120},
        ]
    )
    out = backtest_strategy(
        signals, prices, BacktestConfig(hold_days=180, transaction_cost_bps=0, filing_delay_days=0)
    )
    # Only one trade, the second signal should be skipped
    assert len(out) == 1
    assert out.iloc[0]["buy_date"] == "2024-01-02"


def test_backtest_actual_hold_days():
    """hold_days should reflect actual days held, not config value."""
    signals = pd.DataFrame(
        [{"ticker": "AAA", "date_screened": "2024-01-02", "passes_screen": True}]
    )
    # Only 90 days of data, so exit is end_of_data
    prices = pd.DataFrame(
        [
            {"ticker": "AAA", "date": "2024-01-02", "close": 100},
            {"ticker": "AAA", "date": "2024-04-01", "close": 115},
        ]
    )
    out = backtest_strategy(
        signals, prices, BacktestConfig(hold_days=180, transaction_cost_bps=0, filing_delay_days=0)
    )
    assert len(out) == 1
    assert out.iloc[0]["reason_exit"] == "end_of_data"
    # Actual days held is 90, not 180
    assert out.iloc[0]["hold_days"] == 90


def test_summarize_cagr_is_time_based():
    """CAGR should be based on calendar time, not trade count."""
    results = pd.DataFrame(
        [
            {"return_pct": 100.0, "buy_date": "2022-01-01", "sell_date": "2024-01-01"},
        ]
    )
    summary = summarize_backtest(results)
    # Equity = 2.0, over ~2 years, CAGR ≈ 2^(1/2) - 1 ≈ 0.4142
    assert summary["cagr"] is not None
    assert 0.40 < summary["cagr"] < 0.42


def test_summarize_empty_results():
    results = pd.DataFrame(columns=["return_pct", "buy_date", "sell_date"])
    summary = summarize_backtest(results)
    assert summary["num_trades"] == 0
    assert summary["cagr"] is None
    assert summary["sharpe_ratio"] is None
