from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests


@dataclass
class FMPConfig:
    api_key: str
    base_url: str = "https://financialmodelingprep.com/api/v3"


class YFinanceFetcher:
    def __init__(self) -> None:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError("yfinance is required for YFinanceFetcher") from exc
        self.yf = yf

    def fetch_prices(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        data = self.yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if data.empty:
            return pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"])

        # yfinance may return MultiIndex columns depending on version/options.
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [
                str(col[0]) if isinstance(col, tuple) else str(col)
                for col in data.columns.to_flat_index()
            ]

        data = data.reset_index()
        data.columns = [str(c).strip() for c in data.columns]
        rename_map = {
            "Date": "date",
            "date": "date",
            "Open": "open",
            "open": "open",
            "High": "high",
            "high": "high",
            "Low": "low",
            "low": "low",
            "Close": "close",
            "close": "close",
            "Adj Close": "adj_close",
            "adj close": "adj_close",
            "Adj_Close": "adj_close",
            "adj_close": "adj_close",
            "Volume": "volume",
            "volume": "volume",
        }
        data = data.rename(columns=rename_map)

        # Ensure expected columns exist so downstream insert is stable.
        for col in ["open", "high", "low", "close", "adj_close", "volume"]:
            if col not in data.columns:
                data[col] = None

        if "date" not in data.columns:
            raise ValueError(f"Could not parse date column for ticker={ticker}")

        data["ticker"] = ticker
        data["date"] = pd.to_datetime(data["date"]).dt.date.astype(str)
        cols = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
        out = data[cols].copy()
        out = out.dropna(subset=["ticker", "date"])
        return out

    def fetch_company_profile(self, ticker: str) -> dict[str, Any]:
        info = self.yf.Ticker(ticker).info
        return {
            "ticker": ticker,
            "name": info.get("longName") or info.get("shortName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
        }

    def fetch_fundamentals(self, ticker: str) -> list[dict[str, Any]]:
        tk = self.yf.Ticker(ticker)
        financials = tk.financials.T
        cashflow = tk.cashflow.T
        balance = tk.balance_sheet.T

        if financials.empty:
            return []

        records: list[dict[str, Any]] = []
        for idx, row in financials.iterrows():
            period_end = pd.to_datetime(idx).date().isoformat()
            cf = cashflow.loc[idx] if idx in cashflow.index else pd.Series(dtype=float)
            bs = balance.loc[idx] if idx in balance.index else pd.Series(dtype=float)

            revenue = row.get("Total Revenue")
            net_income = row.get("Net Income")
            gross_profit = row.get("Gross Profit")
            operating_income = row.get("Operating Income")
            operating_cash_flow = cf.get("Operating Cash Flow")
            free_cash_flow = cf.get("Free Cash Flow")
            total_assets = bs.get("Total Assets")
            total_debt = bs.get("Total Debt")
            shareholder_equity = bs.get("Stockholders Equity")
            current_assets = bs.get("Current Assets")
            current_liabilities = bs.get("Current Liabilities")

            current_ratio = None
            if pd.notna(current_assets) and pd.notna(current_liabilities) and current_liabilities:
                current_ratio = float(current_assets / current_liabilities)

            gross_margin = float(gross_profit / revenue) if pd.notna(gross_profit) and pd.notna(revenue) and revenue else None
            operating_margin = (
                float(operating_income / revenue)
                if pd.notna(operating_income) and pd.notna(revenue) and revenue
                else None
            )
            net_margin = float(net_income / revenue) if pd.notna(net_income) and pd.notna(revenue) and revenue else None
            roe = (
                float(net_income / shareholder_equity)
                if pd.notna(net_income) and pd.notna(shareholder_equity) and shareholder_equity
                else None
            )
            debt_to_equity = (
                float(total_debt / shareholder_equity)
                if pd.notna(total_debt) and pd.notna(shareholder_equity) and shareholder_equity
                else None
            )

            records.append(
                {
                    "ticker": ticker,
                    "period_end": period_end,
                    "period_type": "annual",
                    "revenue": _to_float(revenue),
                    "net_income": _to_float(net_income),
                    "operating_cash_flow": _to_float(operating_cash_flow),
                    "free_cash_flow": _to_float(free_cash_flow),
                    "total_assets": _to_float(total_assets),
                    "total_debt": _to_float(total_debt),
                    "shareholder_equity": _to_float(shareholder_equity),
                    "gross_margin": gross_margin,
                    "operating_margin": operating_margin,
                    "net_margin": net_margin,
                    "roe": roe,
                    "debt_to_equity": debt_to_equity,
                    "current_ratio": current_ratio,
                    "raw_payload": {
                        "financials": row.dropna().to_dict(),
                        "cashflow": cf.dropna().to_dict() if not cf.empty else {},
                        "balance": bs.dropna().to_dict() if not bs.empty else {},
                    },
                }
            )

        return records


class FMPFetcher:
    def __init__(self, config: FMPConfig) -> None:
        self.config = config

    def _get(self, endpoint: str, **params: Any) -> Any:
        params["apikey"] = self.config.api_key
        url = f"{self.config.base_url}/{endpoint}"
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def fetch_company_profile(self, ticker: str) -> dict[str, Any]:
        payload = self._get(f"profile/{ticker}")
        if not payload:
            return {"ticker": ticker, "name": None, "sector": None, "industry": None}
        p0 = payload[0]
        return {
            "ticker": ticker,
            "name": p0.get("companyName"),
            "sector": p0.get("sector"),
            "industry": p0.get("industry"),
        }

    def fetch_income_statement(self, ticker: str, limit: int = 20) -> pd.DataFrame:
        data = self._get(f"income-statement/{ticker}", limit=limit)
        return pd.DataFrame(data)

    def fetch_balance_sheet(self, ticker: str, limit: int = 20) -> pd.DataFrame:
        data = self._get(f"balance-sheet-statement/{ticker}", limit=limit)
        return pd.DataFrame(data)

    def fetch_cashflow_statement(self, ticker: str, limit: int = 20) -> pd.DataFrame:
        data = self._get(f"cash-flow-statement/{ticker}", limit=limit)
        return pd.DataFrame(data)

    def fetch_fundamentals(self, ticker: str, limit: int = 20) -> list[dict[str, Any]]:
        income = self.fetch_income_statement(ticker, limit=limit)
        balance = self.fetch_balance_sheet(ticker, limit=limit)
        cash = self.fetch_cashflow_statement(ticker, limit=limit)

        if income.empty:
            return []

        merged = income.merge(
            balance,
            on=["date", "symbol", "reportedCurrency", "calendarYear", "period"],
            how="left",
            suffixes=("", "_balance"),
        ).merge(
            cash,
            on=["date", "symbol", "reportedCurrency", "calendarYear", "period"],
            how="left",
            suffixes=("", "_cash"),
        )

        rows: list[dict[str, Any]] = []
        for _, row in merged.iterrows():
            revenue = row.get("revenue")
            net_income = row.get("netIncome")
            gross_profit = row.get("grossProfit")
            operating_income = row.get("operatingIncome")
            op_cf = row.get("operatingCashFlow")
            fcf = row.get("freeCashFlow")
            equity = row.get("totalStockholdersEquity")
            total_debt = row.get("totalDebt")
            current_assets = row.get("totalCurrentAssets")
            current_liabilities = row.get("totalCurrentLiabilities")

            rows.append(
                {
                    "ticker": ticker,
                    "period_end": row.get("date"),
                    "period_type": "annual" if row.get("period") == "FY" else "quarterly",
                    "currency": row.get("reportedCurrency"),
                    "revenue": _to_float(revenue),
                    "net_income": _to_float(net_income),
                    "operating_cash_flow": _to_float(op_cf),
                    "free_cash_flow": _to_float(fcf),
                    "total_assets": _to_float(row.get("totalAssets")),
                    "total_debt": _to_float(total_debt),
                    "shareholder_equity": _to_float(equity),
                    "eps": _to_float(row.get("eps")),
                    "gross_margin": _ratio(gross_profit, revenue),
                    "operating_margin": _ratio(operating_income, revenue),
                    "net_margin": _ratio(net_income, revenue),
                    "roe": _ratio(net_income, equity),
                    "debt_to_equity": _ratio(total_debt, equity),
                    "current_ratio": _ratio(current_assets, current_liabilities),
                    "raw_payload": row.to_dict(),
                }
            )

        return rows


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _ratio(numerator: Any, denominator: Any) -> float | None:
    num = _to_float(numerator)
    den = _to_float(denominator)
    if num is None or den is None or abs(den) < 1e-12:
        return None
    return num / den
