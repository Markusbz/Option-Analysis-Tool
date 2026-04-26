"""
Unit tests for the MarketDataFetcher.

All yfinance calls are mocked so tests never hit the network.
Tests focus on:
- Correct parsing of spot price, expirations, option chains
- IV cleaning logic (NaN, zeros, outliers, interpolation)
- Risk-free rate fallback
- Dividend yield fallback
- Custom MarketDataError propagation
"""

from unittest.mock import MagicMock, patch, PropertyMock
from collections import namedtuple

import numpy as np
import pandas as pd
import pytest

from engine.data_fetcher import MarketDataFetcher, MarketDataError


# ===================================================================
# Helpers — mock data builders
# ===================================================================

def _make_option_df(strikes, ivs, bids=None, asks=None, volumes=None, ois=None):
    """Build a DataFrame mimicking yfinance option chain columns."""
    n = len(strikes)
    return pd.DataFrame({
        "strike": strikes,
        "impliedVolatility": ivs,
        "bid": bids if bids is not None else [1.0] * n,
        "ask": asks if asks is not None else [2.0] * n,
        "volume": volumes if volumes is not None else [100] * n,
        "openInterest": ois if ois is not None else [500] * n,
    })


OptionChain = namedtuple("OptionChain", ["calls", "puts"])


# ===================================================================
# Spot price
# ===================================================================

class TestFetchSpot:
    @patch("engine.data_fetcher.yf.Ticker")
    def test_returns_regular_market_price(self, mock_ticker_cls):
        mock_tk = MagicMock()
        mock_tk.info = {"regularMarketPrice": 450.50, "previousClose": 449.0}
        mock_ticker_cls.return_value = mock_tk

        result = MarketDataFetcher.fetch_spot("SPY")
        assert result == 450.50

    @patch("engine.data_fetcher.yf.Ticker")
    def test_falls_back_to_previous_close(self, mock_ticker_cls):
        mock_tk = MagicMock()
        mock_tk.info = {"regularMarketPrice": None, "previousClose": 449.0}
        mock_ticker_cls.return_value = mock_tk

        result = MarketDataFetcher.fetch_spot("SPY")
        assert result == 449.0

    @patch("engine.data_fetcher.yf.Ticker")
    def test_raises_on_missing_price(self, mock_ticker_cls):
        mock_tk = MagicMock()
        mock_tk.info = {}
        mock_ticker_cls.return_value = mock_tk

        with pytest.raises(MarketDataError, match="spot price"):
            MarketDataFetcher.fetch_spot("INVALID")

    @patch("engine.data_fetcher.yf.Ticker")
    def test_raises_on_network_error(self, mock_ticker_cls):
        mock_ticker_cls.side_effect = ConnectionError("timeout")

        with pytest.raises(MarketDataError, match="Failed to fetch spot"):
            MarketDataFetcher.fetch_spot("SPY")


# ===================================================================
# Expirations
# ===================================================================

class TestFetchExpirations:
    @patch("engine.data_fetcher.yf.Ticker")
    def test_returns_expirations(self, mock_ticker_cls):
        mock_tk = MagicMock()
        mock_tk.options = ("2026-05-16", "2026-06-20", "2026-09-18")
        mock_ticker_cls.return_value = mock_tk

        result = MarketDataFetcher.fetch_expirations("SPY")
        assert result == ["2026-05-16", "2026-06-20", "2026-09-18"]

    @patch("engine.data_fetcher.yf.Ticker")
    def test_raises_on_empty(self, mock_ticker_cls):
        mock_tk = MagicMock()
        mock_tk.options = ()
        mock_ticker_cls.return_value = mock_tk

        with pytest.raises(MarketDataError, match="No option expirations"):
            MarketDataFetcher.fetch_expirations("INVALID")


# ===================================================================
# Option chain — basic parsing
# ===================================================================

class TestFetchChain:
    @patch("engine.data_fetcher.yf.Ticker")
    def test_basic_chain_parsing(self, mock_ticker_cls):
        calls = _make_option_df([95, 100, 105], [0.20, 0.22, 0.25])
        puts = _make_option_df([95, 100, 105], [0.21, 0.23, 0.26])

        mock_tk = MagicMock()
        mock_tk.option_chain.return_value = OptionChain(calls=calls, puts=puts)
        mock_ticker_cls.return_value = mock_tk

        chain = MarketDataFetcher.fetch_chain("SPY", "2026-06-20")

        assert len(chain) == 6
        assert set(chain.columns) >= {
            "strike", "option_type", "bid", "ask", "mid", "iv", "volume", "openInterest"
        }
        assert set(chain["option_type"].unique()) == {"call", "put"}
        assert (chain["mid"] > 0).all()

    @patch("engine.data_fetcher.yf.Ticker")
    def test_raises_on_empty_chain(self, mock_ticker_cls):
        mock_tk = MagicMock()
        mock_tk.option_chain.return_value = OptionChain(
            calls=pd.DataFrame(), puts=pd.DataFrame()
        )
        mock_ticker_cls.return_value = mock_tk

        with pytest.raises(MarketDataError, match="empty"):
            MarketDataFetcher.fetch_chain("SPY", "2026-06-20")


# ===================================================================
# IV cleaning
# ===================================================================

class TestIVCleaning:
    @patch("engine.data_fetcher.yf.Ticker")
    def test_nan_iv_interpolated(self, mock_ticker_cls):
        """NaN IVs should be filled via linear interpolation."""
        calls = _make_option_df(
            strikes=[90, 95, 100, 105, 110],
            ivs=[0.25, np.nan, np.nan, np.nan, 0.35],
        )
        puts = _make_option_df([100], [0.30])

        mock_tk = MagicMock()
        mock_tk.option_chain.return_value = OptionChain(calls=calls, puts=puts)
        mock_ticker_cls.return_value = mock_tk

        chain = MarketDataFetcher.fetch_chain("SPY", "2026-06-20")
        call_chain = chain[chain["option_type"] == "call"].sort_values("strike")

        # All IVs should be finite and non-NaN
        assert call_chain["iv"].isna().sum() == 0
        assert (call_chain["iv"] > 0).all()
        # Middle strikes should be interpolated between 0.25 and 0.35
        mid_iv = call_chain.iloc[2]["iv"]
        assert 0.25 < mid_iv < 0.35

    @patch("engine.data_fetcher.yf.Ticker")
    def test_zero_iv_treated_as_nan(self, mock_ticker_cls):
        """IVs of 0.0 should be treated as missing and interpolated."""
        calls = _make_option_df(
            strikes=[95, 100, 105],
            ivs=[0.20, 0.0, 0.30],
        )
        puts = _make_option_df([100], [0.25])

        mock_tk = MagicMock()
        mock_tk.option_chain.return_value = OptionChain(calls=calls, puts=puts)
        mock_ticker_cls.return_value = mock_tk

        chain = MarketDataFetcher.fetch_chain("SPY", "2026-06-20")
        call_chain = chain[chain["option_type"] == "call"].sort_values("strike")

        # Middle IV should be interpolated to ~0.25
        mid_iv = call_chain.iloc[1]["iv"]
        assert 0.20 <= mid_iv <= 0.30

    @patch("engine.data_fetcher.yf.Ticker")
    def test_extreme_iv_clamped(self, mock_ticker_cls):
        """IVs above 500% or below 1% should be clamped."""
        calls = _make_option_df(
            strikes=[95, 100, 105],
            ivs=[0.005, 0.25, 8.0],  # 0.5%, 25%, 800%
        )
        puts = _make_option_df([100], [0.25])

        mock_tk = MagicMock()
        mock_tk.option_chain.return_value = OptionChain(calls=calls, puts=puts)
        mock_ticker_cls.return_value = mock_tk

        chain = MarketDataFetcher.fetch_chain("SPY", "2026-06-20")
        call_chain = chain[chain["option_type"] == "call"].sort_values("strike")

        # 0.005 → clamped to 0.01; 8.0 → clamped to 5.0
        assert call_chain.iloc[0]["iv"] == pytest.approx(0.01, abs=1e-6)
        assert call_chain.iloc[2]["iv"] == pytest.approx(5.0, abs=1e-6)

    @patch("engine.data_fetcher.yf.Ticker")
    def test_zero_bid_ask_rows_removed(self, mock_ticker_cls):
        """Rows with both bid=0 and ask=0 should be dropped."""
        calls = _make_option_df(
            strikes=[95, 100, 105],
            ivs=[0.20, 0.25, 0.30],
            bids=[1.0, 0.0, 2.0],
            asks=[2.0, 0.0, 3.0],
        )
        puts = _make_option_df([100], [0.25])

        mock_tk = MagicMock()
        mock_tk.option_chain.return_value = OptionChain(calls=calls, puts=puts)
        mock_ticker_cls.return_value = mock_tk

        chain = MarketDataFetcher.fetch_chain("SPY", "2026-06-20")
        call_chain = chain[chain["option_type"] == "call"]

        # Strike 100 (bid=0, ask=0) should be removed
        assert 100 not in call_chain["strike"].values
        assert len(call_chain) == 2

    @patch("engine.data_fetcher.yf.Ticker")
    def test_volume_filter(self, mock_ticker_cls):
        """Rows below min_volume should be filtered."""
        calls = _make_option_df(
            strikes=[95, 100, 105],
            ivs=[0.20, 0.25, 0.30],
            volumes=[0, 50, 200],
        )
        puts = _make_option_df([100], [0.25], volumes=[100])

        mock_tk = MagicMock()
        mock_tk.option_chain.return_value = OptionChain(calls=calls, puts=puts)
        mock_ticker_cls.return_value = mock_tk

        chain = MarketDataFetcher.fetch_chain("SPY", "2026-06-20", min_volume=10)
        call_chain = chain[chain["option_type"] == "call"]

        assert 95 not in call_chain["strike"].values  # volume=0 filtered
        assert len(call_chain) == 2


# ===================================================================
# Risk-free rate
# ===================================================================

class TestRiskFreeRate:
    @patch("engine.data_fetcher.yf.Ticker")
    def test_returns_irx_divided_by_100(self, mock_ticker_cls):
        mock_tk = MagicMock()
        mock_tk.history.return_value = pd.DataFrame({"Close": [4.25, 4.30]})
        mock_ticker_cls.return_value = mock_tk

        rate = MarketDataFetcher.get_risk_free_rate()
        assert rate == pytest.approx(0.0430, abs=1e-4)

    @patch("engine.data_fetcher.yf.Ticker")
    def test_falls_back_on_empty_history(self, mock_ticker_cls):
        mock_tk = MagicMock()
        mock_tk.history.return_value = pd.DataFrame()
        mock_ticker_cls.return_value = mock_tk

        rate = MarketDataFetcher.get_risk_free_rate(default=0.05)
        assert rate == 0.05

    @patch("engine.data_fetcher.yf.Ticker")
    def test_falls_back_on_exception(self, mock_ticker_cls):
        mock_ticker_cls.side_effect = ConnectionError("timeout")

        rate = MarketDataFetcher.get_risk_free_rate(default=0.04)
        assert rate == 0.04


# ===================================================================
# Dividend yield
# ===================================================================

class TestDividendYield:
    @patch("engine.data_fetcher.yf.Ticker")
    def test_returns_trailing_yield(self, mock_ticker_cls):
        mock_tk = MagicMock()
        mock_tk.info = {"trailingAnnualDividendYield": 0.0132}
        mock_ticker_cls.return_value = mock_tk

        q = MarketDataFetcher.get_dividend_yield("SPY")
        assert q == pytest.approx(0.0132, abs=1e-6)

    @patch("engine.data_fetcher.yf.Ticker")
    def test_falls_back_on_none(self, mock_ticker_cls):
        mock_tk = MagicMock()
        mock_tk.info = {"trailingAnnualDividendYield": None}
        mock_ticker_cls.return_value = mock_tk

        q = MarketDataFetcher.get_dividend_yield("TSLA")
        assert q == 0.0

    @patch("engine.data_fetcher.yf.Ticker")
    def test_falls_back_on_missing_key(self, mock_ticker_cls):
        mock_tk = MagicMock()
        mock_tk.info = {}
        mock_ticker_cls.return_value = mock_tk

        q = MarketDataFetcher.get_dividend_yield("BTC-USD", default=0.0)
        assert q == 0.0

    @patch("engine.data_fetcher.yf.Ticker")
    def test_falls_back_on_exception(self, mock_ticker_cls):
        mock_ticker_cls.side_effect = ConnectionError("timeout")

        q = MarketDataFetcher.get_dividend_yield("SPY", default=0.0)
        assert q == 0.0
