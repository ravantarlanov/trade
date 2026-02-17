import pandas as pd

from stock_system.metrics import add_growth_windows, compute_metrics, pct_change, safe_div


def test_safe_div_and_pct_change():
    assert safe_div(10, 2) == 5
    assert safe_div(10, 0) is None
    assert pct_change(120, 100) == 0.2
    assert pct_change(120, 0) is None


def test_compute_metrics_basic():
    current = {
        "revenue": 120,
        "net_income": 24,
        "operating_cash_flow": 30,
        "total_assets": 200,
        "total_debt": 40,
        "shareholder_equity": 80,
        "gross_profit": 60,
        "operating_income": 30,
        "free_cash_flow": 20,
        "eps": 2.0,
        "price": 40,
        "market_cap": 1000,
    }
    previous = {
        "revenue": 100,
        "net_income": 20,
        "operating_cash_flow": 25,
        "eps": 1.6,
    }
    m = compute_metrics(current, previous)

    assert round(m["revenue_growth_1y"], 4) == 0.2
    assert round(m["earnings_growth_1y"], 4) == 0.2
    assert round(m["net_margin"], 4) == 0.2
    assert round(m["roe"], 4) == 0.3
    assert round(m["debt_to_equity"], 4) == 0.5
    assert round(m["pe_ratio"], 4) == 20.0


def test_add_growth_windows():
    df = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA", "BBB", "BBB"],
            "period_end": ["2020-12-31", "2021-12-31", "2022-12-31", "2021-12-31", "2022-12-31"],
            "revenue": [100, 120, 150, 80, 88],
            "net_income": [10, 12, 16, 8, 9],
            "eps": [1.0, 1.2, 1.5, 0.8, 0.9],
        }
    )

    out = add_growth_windows(df)
    aaa_2022 = out[(out["ticker"] == "AAA") & (out["period_end"] == "2022-12-31")].iloc[0]
    assert round(aaa_2022["revenue_growth_1y"], 4) == 0.25
