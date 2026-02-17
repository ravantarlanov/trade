import json
import tempfile
from pathlib import Path

import pytest

from stock_system.db import Database


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        database = Database(str(db_path))
        database.initialize()
        yield database


def test_initialize_creates_tables(db):
    tables = db.query_df(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    table_names = set(tables["name"].tolist())
    assert {"companies", "financial_metrics", "stock_prices", "screening_results", "backtest_results"}.issubset(
        table_names
    )


def test_upsert_companies_and_query(db):
    db.upsert_companies([{"ticker": "AAPL", "name": "Apple", "sector": "Tech", "industry": "Hardware"}])
    result = db.query_df("SELECT * FROM companies WHERE ticker = ?", ("AAPL",))
    assert len(result) == 1
    assert result.iloc[0]["name"] == "Apple"

    # Upsert same ticker with updated name
    db.upsert_companies([{"ticker": "AAPL", "name": "Apple Inc.", "sector": "Tech", "industry": "Hardware"}])
    result = db.query_df("SELECT * FROM companies WHERE ticker = ?", ("AAPL",))
    assert len(result) == 1
    assert result.iloc[0]["name"] == "Apple Inc."


def test_upsert_stock_prices(db):
    db.upsert_companies([{"ticker": "AAPL", "name": "Apple", "sector": "Tech", "industry": "Hardware"}])
    db.upsert_stock_prices([
        {"ticker": "AAPL", "date": "2024-01-02", "open": 100, "high": 105, "low": 99, "close": 103, "adj_close": 103, "volume": 1000},
    ])
    result = db.query_df("SELECT * FROM stock_prices WHERE ticker = ?", ("AAPL",))
    assert len(result) == 1
    assert result.iloc[0]["close"] == 103


def test_upsert_screening_results(db):
    db.upsert_companies([{"ticker": "AAPL", "name": "Apple", "sector": "Tech", "industry": "Hardware"}])
    db.upsert_screening_results([{
        "ticker": "AAPL",
        "date_screened": "2024-01-31",
        "score": 5,
        "passes_screen": True,
        "criteria_met": ["revenue_growth_1y", "net_margin"],
        "metrics": {"revenue_growth_1y": 0.3},
    }])
    result = db.query_df("SELECT * FROM screening_results WHERE ticker = ?", ("AAPL",))
    assert len(result) == 1
    assert result.iloc[0]["score"] == 5


def test_upsert_backtest_results(db):
    db.upsert_companies([{"ticker": "AAPL", "name": "Apple", "sector": "Tech", "industry": "Hardware"}])
    db.upsert_backtest_results([{
        "ticker": "AAPL",
        "buy_date": "2024-01-02",
        "sell_date": "2024-07-01",
        "hold_days": 181,
        "buy_price": 100.0,
        "sell_price": 120.0,
        "return_pct": 20.0,
        "reason_exit": "time_exit",
        "metrics_at_purchase": {"revenue_growth_1y": 0.3},
    }])
    result = db.query_df("SELECT * FROM backtest_results WHERE ticker = ?", ("AAPL",))
    assert len(result) == 1
    assert result.iloc[0]["return_pct"] == 20.0
    payload = json.loads(result.iloc[0]["metrics_at_purchase"])
    assert payload["revenue_growth_1y"] == 0.3


def test_foreign_key_enforcement(db):
    """Inserting a price for a non-existent company should fail with FK enforcement."""
    with pytest.raises(Exception):
        db.upsert_stock_prices([
            {"ticker": "NONEXISTENT", "date": "2024-01-02", "open": 100, "high": 105, "low": 99, "close": 103, "adj_close": 103, "volume": 1000},
        ])
