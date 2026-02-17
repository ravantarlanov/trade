from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS companies (
    ticker TEXT PRIMARY KEY,
    name TEXT,
    sector TEXT,
    industry TEXT,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS financial_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    period_end TEXT NOT NULL,
    period_type TEXT NOT NULL,
    currency TEXT,
    revenue REAL,
    net_income REAL,
    operating_cash_flow REAL,
    free_cash_flow REAL,
    total_assets REAL,
    total_debt REAL,
    shareholder_equity REAL,
    eps REAL,
    gross_margin REAL,
    operating_margin REAL,
    net_margin REAL,
    roe REAL,
    debt_to_equity REAL,
    current_ratio REAL,
    pe_ratio REAL,
    ps_ratio REAL,
    employee_count REAL,
    raw_payload TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, period_end, period_type)
);

CREATE TABLE IF NOT EXISTS stock_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    adj_close REAL,
    volume REAL,
    market_cap REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, date)
);

CREATE TABLE IF NOT EXISTS screening_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date_screened TEXT NOT NULL,
    score INTEGER NOT NULL,
    passes_screen INTEGER NOT NULL,
    criteria_met TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, date_screened)
);

CREATE TABLE IF NOT EXISTS backtest_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    buy_date TEXT NOT NULL,
    sell_date TEXT NOT NULL,
    hold_days INTEGER NOT NULL,
    buy_price REAL NOT NULL,
    sell_price REAL NOT NULL,
    return_pct REAL NOT NULL,
    reason_exit TEXT,
    metrics_at_purchase TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, buy_date, sell_date, hold_days)
);
"""


class Database:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()

    def upsert_companies(self, companies: list[dict[str, Any]]) -> None:
        if not companies:
            return
        sql = """
        INSERT INTO companies (ticker, name, sector, industry)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            name=excluded.name,
            sector=excluded.sector,
            industry=excluded.industry,
            updated_at=CURRENT_TIMESTAMP
        """
        rows = [
            (
                c.get("ticker"),
                c.get("name"),
                c.get("sector"),
                c.get("industry"),
            )
            for c in companies
        ]
        with self.connect() as conn:
            conn.executemany(sql, rows)
            conn.commit()

    def upsert_financial_metrics(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        sql = """
        INSERT INTO financial_metrics (
            ticker, period_end, period_type, currency,
            revenue, net_income, operating_cash_flow, free_cash_flow,
            total_assets, total_debt, shareholder_equity, eps,
            gross_margin, operating_margin, net_margin, roe,
            debt_to_equity, current_ratio, pe_ratio, ps_ratio,
            employee_count, raw_payload
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, period_end, period_type) DO UPDATE SET
            currency=excluded.currency,
            revenue=excluded.revenue,
            net_income=excluded.net_income,
            operating_cash_flow=excluded.operating_cash_flow,
            free_cash_flow=excluded.free_cash_flow,
            total_assets=excluded.total_assets,
            total_debt=excluded.total_debt,
            shareholder_equity=excluded.shareholder_equity,
            eps=excluded.eps,
            gross_margin=excluded.gross_margin,
            operating_margin=excluded.operating_margin,
            net_margin=excluded.net_margin,
            roe=excluded.roe,
            debt_to_equity=excluded.debt_to_equity,
            current_ratio=excluded.current_ratio,
            pe_ratio=excluded.pe_ratio,
            ps_ratio=excluded.ps_ratio,
            employee_count=excluded.employee_count,
            raw_payload=excluded.raw_payload
        """
        rows = []
        for r in records:
            rows.append(
                (
                    r.get("ticker"),
                    r.get("period_end"),
                    r.get("period_type", "annual"),
                    r.get("currency"),
                    r.get("revenue"),
                    r.get("net_income"),
                    r.get("operating_cash_flow"),
                    r.get("free_cash_flow"),
                    r.get("total_assets"),
                    r.get("total_debt"),
                    r.get("shareholder_equity"),
                    r.get("eps"),
                    r.get("gross_margin"),
                    r.get("operating_margin"),
                    r.get("net_margin"),
                    r.get("roe"),
                    r.get("debt_to_equity"),
                    r.get("current_ratio"),
                    r.get("pe_ratio"),
                    r.get("ps_ratio"),
                    r.get("employee_count"),
                    json.dumps(r.get("raw_payload", {})),
                )
            )
        with self.connect() as conn:
            conn.executemany(sql, rows)
            conn.commit()

    def upsert_stock_prices(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        sql = """
        INSERT INTO stock_prices (
            ticker, date, open, high, low, close, adj_close, volume, market_cap
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, date) DO UPDATE SET
            open=excluded.open,
            high=excluded.high,
            low=excluded.low,
            close=excluded.close,
            adj_close=excluded.adj_close,
            volume=excluded.volume,
            market_cap=excluded.market_cap
        """
        rows = []
        for r in records:
            ticker = r.get("ticker")
            date = r.get("date")
            if ticker is None or date is None:
                continue
            rows.append(
                (
                    ticker,
                    date,
                    r.get("open"),
                    r.get("high"),
                    r.get("low"),
                    r.get("close"),
                    r.get("adj_close"),
                    r.get("volume"),
                    r.get("market_cap"),
                )
            )
        if not rows:
            return
        with self.connect() as conn:
            conn.executemany(sql, rows)
            conn.commit()

    def upsert_screening_results(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        sql = """
        INSERT INTO screening_results (
            ticker, date_screened, score, passes_screen, criteria_met, metrics_json
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, date_screened) DO UPDATE SET
            score=excluded.score,
            passes_screen=excluded.passes_screen,
            criteria_met=excluded.criteria_met,
            metrics_json=excluded.metrics_json
        """
        rows = [
            (
                r["ticker"],
                r["date_screened"],
                int(r["score"]),
                int(bool(r["passes_screen"])),
                json.dumps(r.get("criteria_met", [])),
                json.dumps(r.get("metrics", {})),
            )
            for r in records
        ]
        with self.connect() as conn:
            conn.executemany(sql, rows)
            conn.commit()

    def upsert_backtest_results(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        sql = """
        INSERT INTO backtest_results (
            ticker, buy_date, sell_date, hold_days, buy_price, sell_price,
            return_pct, reason_exit, metrics_at_purchase
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, buy_date, sell_date, hold_days) DO UPDATE SET
            buy_price=excluded.buy_price,
            sell_price=excluded.sell_price,
            return_pct=excluded.return_pct,
            reason_exit=excluded.reason_exit,
            metrics_at_purchase=excluded.metrics_at_purchase
        """
        rows = [
            (
                r["ticker"],
                r["buy_date"],
                r["sell_date"],
                int(r["hold_days"]),
                float(r["buy_price"]),
                float(r["sell_price"]),
                float(r["return_pct"]),
                r.get("reason_exit"),
                json.dumps(r.get("metrics_at_purchase", {})),
            )
            for r in records
        ]
        with self.connect() as conn:
            conn.executemany(sql, rows)
            conn.commit()

    def query_df(self, query: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
        with self.connect() as conn:
            return pd.read_sql_query(query, conn, params=params)
