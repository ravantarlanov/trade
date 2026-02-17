from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from .analysis import create_core_plots, export_core_reports
from .backtest import BacktestConfig, backtest_strategy, summarize_backtest
from .db import Database
from .fetchers import FMPConfig, FMPFetcher, YFinanceFetcher
from .metrics import add_growth_windows, add_revenue_acceleration
from .screening import ScreeningCriteria, screen_universe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stock screening and backtesting system")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init-db", help="Initialize SQLite schema")
    p_init.add_argument("--db-path", required=True)

    p_prices = sub.add_parser("fetch-prices", help="Fetch historical prices using yfinance")
    p_prices.add_argument("--db-path", required=True)
    p_prices.add_argument("--tickers", nargs="+", required=True)
    p_prices.add_argument("--start", required=True)
    p_prices.add_argument("--end", required=True)

    p_fund = sub.add_parser("fetch-fundamentals", help="Fetch fundamentals from yfinance or FMP")
    p_fund.add_argument("--db-path", required=True)
    p_fund.add_argument("--tickers", nargs="+", required=True)
    p_fund.add_argument("--source", choices=["yfinance", "fmp"], default="yfinance")

    p_screen = sub.add_parser("screen", help="Run screening on latest financial rows")
    p_screen.add_argument("--db-path", required=True)
    p_screen.add_argument("--as-of", required=True)

    p_backtest = sub.add_parser("backtest", help="Backtest from stored screening signals")
    p_backtest.add_argument("--db-path", required=True)
    p_backtest.add_argument("--hold-days", type=int, default=180)
    p_backtest.add_argument("--start")
    p_backtest.add_argument("--end")
    p_backtest.add_argument("--transaction-cost-bps", type=float, default=0.0)
    p_backtest.add_argument("--filing-delay-days", type=int, default=45,
                            help="Days after signal to delay buy (avoids look-ahead bias)")

    p_analyze = sub.add_parser("analyze", help="Export CSV reports and plots")
    p_analyze.add_argument("--db-path", required=True)
    p_analyze.add_argument("--output-dir", required=True)

    return parser.parse_args()


def cmd_init_db(db: Database) -> None:
    db.initialize()
    print(f"Initialized DB schema: {db.db_path}")


def cmd_fetch_prices(db: Database, tickers: list[str], start: str, end: str) -> None:
    fetcher = YFinanceFetcher()
    total_rows = 0
    for ticker in tickers:
        profile = fetcher.fetch_company_profile(ticker)
        db.upsert_companies([profile])

        prices = fetcher.fetch_prices(ticker, start=start, end=end)
        db.upsert_stock_prices(prices.to_dict(orient="records"))
        total_rows += len(prices)
        print(f"{ticker}: inserted/updated {len(prices)} price rows")

    print(f"Completed price fetch. Total rows: {total_rows}")


def cmd_fetch_fundamentals(
    db: Database,
    tickers: list[str],
    source: str,
    fmp_api_key: str | None,
) -> None:
    if source == "yfinance":
        fetcher = YFinanceFetcher()
    else:
        if not fmp_api_key:
            raise ValueError("FMP API key required for source=fmp")
        fetcher = FMPFetcher(FMPConfig(api_key=fmp_api_key))

    total_rows = 0
    for ticker in tickers:
        profile = fetcher.fetch_company_profile(ticker)
        db.upsert_companies([profile])

        records = fetcher.fetch_fundamentals(ticker)
        db.upsert_financial_metrics(records)
        total_rows += len(records)
        print(f"{ticker}: inserted/updated {len(records)} fundamental rows")

    print(f"Completed fundamentals fetch. Total rows: {total_rows}")


def cmd_screen(db: Database, as_of: str) -> None:
    query = """
    SELECT *
    FROM financial_metrics
    WHERE period_end <= ?
    ORDER BY ticker, period_end
    """
    df = db.query_df(query, (as_of,))
    if df.empty:
        print("No financial rows available before as-of date")
        return

    enriched = add_growth_windows(df)
    enriched = add_revenue_acceleration(enriched)

    latest = enriched.groupby("ticker", as_index=False).tail(1).copy()
    latest["date_screened"] = as_of

    criteria = ScreeningCriteria()
    screen_df = screen_universe(latest, date_col="date_screened", criteria=criteria)
    records = screen_df.to_dict(orient="records")
    db.upsert_screening_results(records)

    passed = int(screen_df["passes_screen"].sum())
    print(f"Screened {len(screen_df)} stocks; passed: {passed}")


def cmd_backtest(
    db: Database,
    hold_days: int,
    start: str | None,
    end: str | None,
    transaction_cost_bps: float,
    filing_delay_days: int = 45,
) -> None:
    q_screen = "SELECT ticker, date_screened, passes_screen, metrics_json FROM screening_results"
    filters = []
    params: list[str] = []
    if start:
        filters.append("date_screened >= ?")
        params.append(start)
    if end:
        filters.append("date_screened <= ?")
        params.append(end)
    if filters:
        q_screen += " WHERE " + " AND ".join(filters)

    signals = db.query_df(q_screen, tuple(params))
    if signals.empty:
        print("No screening signals found for backtest")
        return

    metrics_expanded = signals["metrics_json"].apply(lambda s: json.loads(s) if s else {})
    metrics_df = pd.json_normalize(metrics_expanded)
    signals = pd.concat([signals.drop(columns=["metrics_json"]), metrics_df], axis=1)

    tickers = signals["ticker"].unique().tolist()
    placeholders = ",".join("?" for _ in tickers)
    prices = db.query_df(
        f"SELECT ticker, date, close FROM stock_prices WHERE ticker IN ({placeholders})",
        tuple(tickers),
    )
    results = backtest_strategy(
        signals,
        prices,
        config=BacktestConfig(
            hold_days=hold_days,
            transaction_cost_bps=transaction_cost_bps,
            filing_delay_days=filing_delay_days,
        ),
    )

    if results.empty:
        print("No backtest trades generated")
        return

    if not metrics_df.empty:
        metrics_cols = [c for c in metrics_df.columns if c in signals.columns]
        merged = results.merge(
            signals[["ticker", "date_screened", *metrics_cols]],
            left_on=["ticker", "signal_date"],
            right_on=["ticker", "date_screened"],
            how="left",
        )
    else:
        merged = results

    db_rows = []
    for _, row in merged.iterrows():
        metrics = {
            k: row[k]
            for k in merged.columns
            if k
            not in {
                "ticker",
                "buy_date",
                "sell_date",
                "hold_days",
                "buy_price",
                "sell_price",
                "return_pct",
                "reason_exit",
                "signal_date",
                "date_screened",
            }
        }
        db_rows.append(
            {
                "ticker": row["ticker"],
                "buy_date": row["buy_date"],
                "sell_date": row["sell_date"],
                "hold_days": row["hold_days"],
                "buy_price": row["buy_price"],
                "sell_price": row["sell_price"],
                "return_pct": row["return_pct"],
                "reason_exit": row.get("reason_exit"),
                "metrics_at_purchase": metrics,
            }
        )

    db.upsert_backtest_results(db_rows)
    summary = summarize_backtest(results)
    print("Backtest summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


def cmd_analyze(db: Database, output_dir: str) -> None:
    q = """
    SELECT br.*, json_extract(br.metrics_at_purchase, '$.revenue_growth_1y') AS revenue_growth_1y,
           json_extract(br.metrics_at_purchase, '$.pe_ratio') AS pe_ratio
    FROM backtest_results br
    """
    results = db.query_df(q)
    if results.empty:
        print("No backtest results available for analysis")
        return

    output_path = Path(output_dir)
    csv_paths = export_core_reports(results, output_path)
    plot_paths = create_core_plots(results, output_path / "visualizations")

    print("Generated CSV reports:")
    for key, path in csv_paths.items():
        print(f"  {key}: {path}")

    print("Generated plots:")
    for key, path in plot_paths.items():
        print(f"  {key}: {path}")


def main() -> None:
    args = parse_args()
    db = Database(args.db_path)

    if args.command == "init-db":
        cmd_init_db(db)
    elif args.command == "fetch-prices":
        cmd_fetch_prices(db, args.tickers, args.start, args.end)
    elif args.command == "fetch-fundamentals":
        cmd_fetch_fundamentals(db, args.tickers, args.source, os.getenv("FMP_API_KEY"))
    elif args.command == "screen":
        cmd_screen(db, args.as_of)
    elif args.command == "backtest":
        cmd_backtest(db, args.hold_days, args.start, args.end, args.transaction_cost_bps, args.filing_delay_days)
    elif args.command == "analyze":
        cmd_analyze(db, args.output_dir)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
