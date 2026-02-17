from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class ScreeningCriteria:
    min_revenue_growth_1y: float = 0.15
    min_earnings_growth_1y: float = 0.15
    min_profit_margin: float = 0.10
    max_pe_ratio: float = 30.0
    max_debt_to_equity: float = 2.0
    min_free_cash_flow: float = 0.0
    min_score: int = 5


def _meets(metric_value: float | None, op: str, threshold: float) -> bool:
    if metric_value is None or pd.isna(metric_value):
        return False
    if op == ">":
        return metric_value > threshold
    if op == ">=":
        return metric_value >= threshold
    if op == "<":
        return metric_value < threshold
    if op == "<=":
        return metric_value <= threshold
    raise ValueError(f"Unsupported operator: {op}")


def score_metrics(metrics: dict[str, Any], criteria: ScreeningCriteria) -> tuple[int, list[str]]:
    checks = {
        "revenue_growth_1y": (
            metrics.get("revenue_growth_1y"),
            ">",
            criteria.min_revenue_growth_1y,
        ),
        "earnings_growth_1y": (
            metrics.get("earnings_growth_1y"),
            ">",
            criteria.min_earnings_growth_1y,
        ),
        "net_margin": (metrics.get("net_margin"), ">", criteria.min_profit_margin),
        "pe_ratio": (metrics.get("pe_ratio"), "<", criteria.max_pe_ratio),
        "debt_to_equity": (
            metrics.get("debt_to_equity"),
            "<",
            criteria.max_debt_to_equity,
        ),
        "free_cash_flow": (
            metrics.get("free_cash_flow"),
            ">",
            criteria.min_free_cash_flow,
        ),
    }

    met = [name for name, (val, op, threshold) in checks.items() if _meets(val, op, threshold)]
    return len(met), met


def screen_stock(
    ticker: str,
    date_screened: str,
    metrics: dict[str, Any],
    criteria: ScreeningCriteria | None = None,
) -> dict[str, Any]:
    criteria = criteria or ScreeningCriteria()
    score, criteria_met = score_metrics(metrics, criteria)
    return {
        "ticker": ticker,
        "date_screened": date_screened,
        "metrics": metrics,
        "score": score,
        "criteria_met": criteria_met,
        "passes_screen": score >= criteria.min_score,
    }


def screen_universe(
    metrics_df: pd.DataFrame,
    date_col: str = "period_end",
    criteria: ScreeningCriteria | None = None,
) -> pd.DataFrame:
    criteria = criteria or ScreeningCriteria()
    if metrics_df.empty:
        return pd.DataFrame(
            columns=["ticker", "date_screened", "score", "criteria_met", "passes_screen"]
        )

    metric_cols = [
        "revenue_growth_1y",
        "earnings_growth_1y",
        "net_margin",
        "pe_ratio",
        "debt_to_equity",
        "free_cash_flow",
    ]

    rows = []
    for _, row in metrics_df.iterrows():
        metrics = {c: row.get(c) for c in metric_cols}
        rows.append(
            screen_stock(
                ticker=row["ticker"],
                date_screened=str(row[date_col]),
                metrics=metrics,
                criteria=criteria,
            )
        )

    return pd.DataFrame(rows)
