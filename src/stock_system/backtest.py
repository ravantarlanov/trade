from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class BacktestConfig:
    hold_days: int = 180
    transaction_cost_bps: float = 0.0


def _normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["ticker", "date"])


def _price_on_or_after(ticker_prices: pd.DataFrame, target_date: pd.Timestamp) -> tuple[pd.Timestamp, float] | None:
    subset = ticker_prices[ticker_prices["date"] >= target_date]
    if subset.empty:
        return None
    row = subset.iloc[0]
    return row["date"], float(row["close"])


def backtest_strategy(
    screen_results: pd.DataFrame,
    prices: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    config = config or BacktestConfig()

    if screen_results.empty or prices.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "signal_date",
                "buy_date",
                "sell_date",
                "hold_days",
                "buy_price",
                "sell_price",
                "return_pct",
                "reason_exit",
            ]
        )

    price_df = _normalize_prices(prices)
    screen_df = screen_results.copy()
    screen_df["date_screened"] = pd.to_datetime(screen_df["date_screened"])

    rows: list[dict[str, Any]] = []
    for _, signal in screen_df.iterrows():
        if not bool(signal.get("passes_screen", True)):
            continue

        ticker = signal["ticker"]
        ticker_prices = price_df[price_df["ticker"] == ticker]
        if ticker_prices.empty:
            continue

        buy_result = _price_on_or_after(ticker_prices, signal["date_screened"])
        if buy_result is None:
            continue

        buy_date, buy_price = buy_result
        target_sell_date = buy_date + timedelta(days=config.hold_days)
        sell_result = _price_on_or_after(ticker_prices, target_sell_date)

        if sell_result is None:
            sell_row = ticker_prices.iloc[-1]
            sell_date = pd.to_datetime(sell_row["date"])
            sell_price = float(sell_row["close"])
            reason_exit = "end_of_data"
        else:
            sell_date, sell_price = sell_result
            reason_exit = "time_exit"

        gross_return = (sell_price - buy_price) / buy_price
        total_cost = 2 * config.transaction_cost_bps / 10000.0
        net_return = gross_return - total_cost

        rows.append(
            {
                "ticker": ticker,
                "signal_date": signal["date_screened"].date().isoformat(),
                "buy_date": buy_date.date().isoformat(),
                "sell_date": sell_date.date().isoformat(),
                "hold_days": config.hold_days,
                "buy_price": buy_price,
                "sell_price": sell_price,
                "return_pct": net_return * 100,
                "reason_exit": reason_exit,
            }
        )

    return pd.DataFrame(rows)


def summarize_backtest(results: pd.DataFrame) -> dict[str, float | int | None]:
    if results.empty:
        return {
            "num_trades": 0,
            "avg_return_pct": None,
            "win_rate": None,
            "best_trade_pct": None,
            "worst_trade_pct": None,
            "sharpe_ratio": None,
            "max_drawdown_pct": None,
            "cagr": None,
            "win_rate_p_value": None,
        }

    r = results["return_pct"].astype(float)
    win_rate = float((r > 0).mean())

    equity = (1 + r / 100.0).cumprod()
    running_max = equity.cummax()
    drawdowns = equity / running_max - 1
    max_drawdown = float(drawdowns.min())

    periods = max(len(r), 1)
    cagr = float(equity.iloc[-1] ** (252 / periods) - 1)

    excess = r / 100.0
    sharpe = None
    if excess.std(ddof=1) > 0:
        sharpe = float((excess.mean() / excess.std(ddof=1)) * np.sqrt(252))

    wins = int((r > 0).sum())
    binom = stats.binomtest(wins, len(r), p=0.5, alternative="greater")

    return {
        "num_trades": int(len(r)),
        "avg_return_pct": float(r.mean()),
        "win_rate": win_rate,
        "best_trade_pct": float(r.max()),
        "worst_trade_pct": float(r.min()),
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_drawdown * 100,
        "cagr": cagr,
        "win_rate_p_value": float(binom.pvalue),
    }
