from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class MetricConfig:
    min_denominator: float = 1e-9


def safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if abs(denominator) < 1e-12:
        return None
    return numerator / denominator


def pct_change(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None:
        return None
    if abs(previous) < 1e-12:
        return None
    return (current - previous) / abs(previous)


def compute_metrics(current: dict[str, Any], previous: dict[str, Any] | None = None) -> dict[str, float | None]:
    previous = previous or {}

    revenue = current.get("revenue")
    net_income = current.get("net_income")
    operating_cash_flow = current.get("operating_cash_flow")
    total_assets = current.get("total_assets")
    total_debt = current.get("total_debt")
    shareholder_equity = current.get("shareholder_equity")
    gross_profit = current.get("gross_profit")
    operating_income = current.get("operating_income")
    market_cap = current.get("market_cap")

    metrics: dict[str, float | None] = {
        "revenue_growth_1y": pct_change(revenue, previous.get("revenue")),
        "earnings_growth_1y": pct_change(net_income, previous.get("net_income")),
        "operating_cash_flow_growth_1y": pct_change(
            operating_cash_flow, previous.get("operating_cash_flow")
        ),
        "gross_margin": safe_div(gross_profit, revenue),
        "operating_margin": safe_div(operating_income, revenue),
        "net_margin": safe_div(net_income, revenue),
        "roe": safe_div(net_income, shareholder_equity),
        "debt_to_equity": safe_div(total_debt, shareholder_equity),
        "asset_turnover": safe_div(revenue, total_assets),
        "free_cash_flow": current.get("free_cash_flow"),
        "pe_ratio": current.get("pe_ratio"),
        "ps_ratio": safe_div(market_cap, revenue),
        "current_ratio": current.get("current_ratio"),
        "eps_growth_1y": pct_change(current.get("eps"), previous.get("eps")),
    }

    if metrics["pe_ratio"] is None:
        price = current.get("price")
        eps = current.get("eps")
        metrics["pe_ratio"] = safe_div(price, eps)

    return metrics


def add_growth_windows(financial_df: pd.DataFrame) -> pd.DataFrame:
    if financial_df.empty:
        return financial_df.copy()

    df = financial_df.copy()
    df = df.sort_values(["ticker", "period_end"])

    for base_col, prefix in [
        ("revenue", "revenue_growth"),
        ("net_income", "earnings_growth"),
        ("eps", "eps_growth"),
    ]:
        df[f"{prefix}_1y"] = df.groupby("ticker")[base_col].pct_change(periods=1)
        df[f"{prefix}_3y"] = df.groupby("ticker")[base_col].pct_change(periods=3)
        df[f"{prefix}_5y"] = df.groupby("ticker")[base_col].pct_change(periods=5)

    return df


def add_revenue_acceleration(financial_df: pd.DataFrame) -> pd.DataFrame:
    if financial_df.empty:
        return financial_df.copy()

    df = financial_df.copy()
    if "revenue_growth_1y" not in df.columns:
        df = add_growth_windows(df)

    df = df.sort_values(["ticker", "period_end"])
    df["revenue_acceleration"] = df.groupby("ticker")["revenue_growth_1y"].diff()
    return df


def correlation_with_returns(df: pd.DataFrame, return_col: str = "return_pct") -> pd.DataFrame:
    if df.empty or return_col not in df.columns:
        return pd.DataFrame(columns=["metric", "correlation"])

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != return_col]
    rows = []
    for col in numeric_cols:
        valid = df[[col, return_col]].dropna()
        if len(valid) < 3:
            corr = np.nan
        else:
            corr = valid[col].corr(valid[return_col])
        rows.append({"metric": col, "correlation": corr})

    return pd.DataFrame(rows).sort_values("correlation", ascending=False)
