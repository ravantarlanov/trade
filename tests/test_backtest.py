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

    out = backtest_strategy(signals, prices, BacktestConfig(hold_days=180, transaction_cost_bps=0))
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

    out = backtest_strategy(signals, prices, BacktestConfig(hold_days=180, transaction_cost_bps=50))
    # Gross 10%, roundtrip cost 1% => net 9%
    assert round(out.iloc[0]["return_pct"], 4) == 9.0


def test_summarize_backtest_values():
    results = pd.DataFrame(
        [
            {"return_pct": 10.0},
            {"return_pct": -5.0},
            {"return_pct": 8.0},
        ]
    )
    summary = summarize_backtest(results)
    assert summary["num_trades"] == 3
    assert round(summary["avg_return_pct"], 4) == 4.3333
    assert round(summary["win_rate"], 4) == 0.6667
