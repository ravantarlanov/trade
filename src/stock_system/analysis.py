from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .backtest import summarize_backtest
from .metrics import correlation_with_returns


def build_summary_statistics(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()

    summary = summarize_backtest(results_df)
    return pd.DataFrame([summary])


def metric_bucket_report(
    results_df: pd.DataFrame,
    metric_col: str,
    return_col: str = "return_pct",
    buckets: int = 3,
) -> pd.DataFrame:
    if results_df.empty or metric_col not in results_df.columns:
        return pd.DataFrame()

    df = results_df[[metric_col, return_col]].dropna().copy()
    if df.empty:
        return pd.DataFrame()

    df["bucket"] = pd.qcut(df[metric_col], q=buckets, duplicates="drop")
    report = (
        df.groupby("bucket", observed=True)[return_col]
        .agg(avg_return="mean", win_rate=lambda x: (x > 0).mean(), count="count")
        .reset_index()
    )
    return report


def export_core_reports(results_df: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    backtest_path = out / "backtest_results.csv"
    results_df.to_csv(backtest_path, index=False)
    paths["backtest_results"] = backtest_path

    summary = build_summary_statistics(results_df)
    summary_path = out / "summary_statistics.csv"
    summary.to_csv(summary_path, index=False)
    paths["summary_statistics"] = summary_path

    corr = correlation_with_returns(results_df, return_col="return_pct")
    corr_path = out / "metric_correlation.csv"
    corr.to_csv(corr_path, index=False)
    paths["metric_correlation"] = corr_path

    return paths


def create_core_plots(results_df: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    paths: dict[str, Path] = {}

    if {"revenue_growth_1y", "return_pct"}.issubset(results_df.columns):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=results_df, x="revenue_growth_1y", y="return_pct")
        plt.title("Revenue Growth vs Return")
        p = out / "scatter_revenue_growth_vs_return.png"
        plt.tight_layout()
        plt.savefig(p)
        plt.close()
        paths["scatter_revenue_growth_vs_return"] = p

    if {"pe_ratio", "return_pct"}.issubset(results_df.columns):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=results_df, x="pe_ratio", y="return_pct")
        plt.title("P/E Ratio vs Return")
        p = out / "scatter_pe_vs_return.png"
        plt.tight_layout()
        plt.savefig(p)
        plt.close()
        paths["scatter_pe_vs_return"] = p

    if "return_pct" in results_df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(results_df["return_pct"].dropna(), bins=30, kde=True)
        plt.title("Return Distribution")
        p = out / "hist_return_distribution.png"
        plt.tight_layout()
        plt.savefig(p)
        plt.close()
        paths["hist_return_distribution"] = p

        plt.figure(figsize=(10, 6))
        cumulative = (1 + results_df["return_pct"].fillna(0) / 100.0).cumprod()
        plt.plot(cumulative.values)
        plt.title("Cumulative Returns")
        plt.xlabel("Trade Number")
        plt.ylabel("Growth of $1")
        p = out / "timeseries_cumulative_returns.png"
        plt.tight_layout()
        plt.savefig(p)
        plt.close()
        paths["timeseries_cumulative_returns"] = p

    return paths
