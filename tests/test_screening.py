from stock_system.screening import ScreeningCriteria, screen_stock


def test_screen_stock_passes_with_high_score():
    criteria = ScreeningCriteria(min_score=5)
    metrics = {
        "revenue_growth_1y": 0.3,
        "earnings_growth_1y": 0.2,
        "net_margin": 0.15,
        "pe_ratio": 20,
        "debt_to_equity": 1.2,
        "free_cash_flow": 1000,
    }
    result = screen_stock("ABC", "2024-01-31", metrics, criteria)
    assert result["score"] == 6
    assert result["passes_screen"] is True


def test_screen_stock_fails_when_insufficient_criteria():
    criteria = ScreeningCriteria(min_score=5)
    metrics = {
        "revenue_growth_1y": 0.05,
        "earnings_growth_1y": 0.02,
        "net_margin": 0.03,
        "pe_ratio": 45,
        "debt_to_equity": 3.5,
        "free_cash_flow": -10,
    }
    result = screen_stock("XYZ", "2024-01-31", metrics, criteria)
    assert result["score"] == 0
    assert result["passes_screen"] is False
