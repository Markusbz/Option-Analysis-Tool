"""
Market data fetcher using yfinance.

Provides a ``MarketDataFetcher`` class that retrieves:
- Current spot / underlying price
- Available option expiration dates
- Full option chains (calls + puts) with IV cleaning
- Risk-free rate (^IRX 13-Week T-Bill)
- Continuous dividend yield

All network calls are wrapped in try/except to raise a custom
``MarketDataError`` that the UI can catch for user-friendly error handling.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class MarketDataError(Exception):
    """Raised when a yfinance network request fails or returns invalid data."""
    pass


# ---------------------------------------------------------------------------
# IV cleaning constants
# ---------------------------------------------------------------------------

_IV_MIN = 0.01    # 1 % — anything below is considered bad data
_IV_MAX = 5.00    # 500 % — anything above is considered an outlier


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

class MarketDataFetcher:
    """Thin wrapper around yfinance with error handling and IV cleaning.

    Usage
    -----
    >>> fetcher = MarketDataFetcher()
    >>> spot = fetcher.fetch_spot("SPY")
    >>> expirations = fetcher.fetch_expirations("SPY")
    >>> chain = fetcher.fetch_chain("SPY", expirations[0])
    >>> r = fetcher.get_risk_free_rate()
    >>> q = fetcher.get_dividend_yield("SPY")
    """

    # ----- spot price ------------------------------------------------------

    @staticmethod
    def fetch_spot(ticker: str) -> float:
        """Fetch the current underlying price.

        Parameters
        ----------
        ticker : str
            Yahoo Finance ticker symbol (e.g. ``'SPY'``, ``'AAPL'``).

        Returns
        -------
        float
            Most recent trade / close price.

        Raises
        ------
        MarketDataError
            If the ticker is invalid or the network request fails.
        """
        try:
            tk = yf.Ticker(ticker)
            info = tk.info
            # Prefer regularMarketPrice, fall back to previousClose
            price = info.get("regularMarketPrice") or info.get("previousClose")
            if price is None:
                raise MarketDataError(
                    f"Could not retrieve spot price for '{ticker}'. "
                    "Check that the ticker symbol is valid."
                )
            return float(price)
        except MarketDataError:
            raise
        except Exception as exc:
            raise MarketDataError(
                f"Failed to fetch spot price for '{ticker}': {exc}"
            ) from exc

    # ----- expirations -----------------------------------------------------

    @staticmethod
    def fetch_expirations(ticker: str) -> List[str]:
        """Fetch available option expiration dates.

        Returns
        -------
        list[str]
            Sorted ISO-format date strings (e.g. ``['2026-05-16', ...]``).

        Raises
        ------
        MarketDataError
            If no expirations are found or the request fails.
        """
        try:
            tk = yf.Ticker(ticker)
            expirations = tk.options
            if not isinstance(expirations, (list, tuple)) or len(expirations) == 0:
                raise MarketDataError(
                    f"No options chain available for '{ticker}' on Yahoo Finance."
                )
            return list(expirations)
        except MarketDataError:
            raise
        except Exception as exc:
            raise MarketDataError(
                f"Failed to fetch expirations for '{ticker}': {exc}"
            ) from exc

    # ----- option chain ----------------------------------------------------

    @classmethod
    def fetch_chain(
        cls,
        ticker: str,
        expiry: str,
        min_volume: int = 0,
        min_open_interest: int = 0,
    ) -> pd.DataFrame:
        """Fetch and clean the option chain for a single expiration.

        Parameters
        ----------
        ticker : str
            Underlying ticker.
        expiry : str
            Expiration date (ISO format, from ``fetch_expirations``).
        min_volume : int
            Drop rows with volume below this threshold.
        min_open_interest : int
            Drop rows with open interest below this threshold.

        Returns
        -------
        pd.DataFrame
            Cleaned chain with columns:
            ``strike, option_type, bid, ask, mid, iv, volume, openInterest``

        Raises
        ------
        MarketDataError
            If the chain cannot be retrieved or is empty after cleaning.
        """
        try:
            tk = yf.Ticker(ticker)
            raw = tk.option_chain(expiry)
        except Exception as exc:
            raise MarketDataError(
                f"Failed to fetch option chain for '{ticker}' exp={expiry}: {exc}"
            ) from exc

        frames = []
        for opt_type, df in [("call", raw.calls), ("put", raw.puts)]:
            if df is None or df.empty:
                continue
            sub = pd.DataFrame({
                "strike": df["strike"],
                "option_type": opt_type,
                "bid": df.get("bid", 0.0),
                "ask": df.get("ask", 0.0),
                "iv": df.get("impliedVolatility", np.nan),
                "volume": df.get("volume", 0),
                "openInterest": df.get("openInterest", 0),
            })
            frames.append(sub)

        if not frames:
            raise MarketDataError(
                f"Option chain is empty for '{ticker}' exp={expiry}."
            )

        chain = pd.concat(frames, ignore_index=True)

        # ---- Cleaning ----
        chain = cls._clean_chain(chain, min_volume, min_open_interest)

        if chain.empty:
            raise MarketDataError(
                f"Option chain for '{ticker}' exp={expiry} is empty "
                "after cleaning (all rows filtered out)."
            )

        # Compute mid price
        chain["mid"] = (chain["bid"] + chain["ask"]) / 2.0

        return chain.reset_index(drop=True)

    @staticmethod
    def _clean_chain(
        chain: pd.DataFrame,
        min_volume: int,
        min_open_interest: int,
    ) -> pd.DataFrame:
        """Clean IV and filter bad rows from an option chain.

        1. Fill zero/NaN IVs per option_type using linear interpolation
           on the strike axis, with forward/backward fill for edges.
        2. Clamp remaining IVs to [_IV_MIN, _IV_MAX].
        3. Drop rows with zero bid AND zero ask (no market).
        4. Apply volume / open-interest filters.
        """
        df = chain.copy()

        # Convert volume / OI to numeric (yfinance sometimes returns NaN)
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
        df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce").fillna(0).astype(int)

        # Replace zero or negative IV with NaN for interpolation
        df.loc[df["iv"] <= 0, "iv"] = np.nan

        # Per-option-type interpolation on strike axis
        for otype in ("call", "put"):
            mask = df["option_type"] == otype
            if mask.sum() < 2:
                continue
            sub = df.loc[mask].sort_values("strike")
            sub["iv"] = (
                sub["iv"]
                .interpolate(method="linear", limit_direction="both")
                .ffill()
                .bfill()
            )
            df.loc[sub.index, "iv"] = sub["iv"]

        # Clamp IV to bounds
        df["iv"] = df["iv"].clip(lower=_IV_MIN, upper=_IV_MAX)

        # Fill any remaining NaN IVs with the median IV of the entire chain
        median_iv = df["iv"].median()
        if pd.isna(median_iv):
            median_iv = 0.20   # last-resort fallback
        df["iv"] = df["iv"].fillna(median_iv)

        # Drop rows where both bid and ask are zero (no market)
        df = df[~((df["bid"] <= 0) & (df["ask"] <= 0))]

        # Volume / OI filters
        if min_volume > 0:
            df = df[df["volume"] >= min_volume]
        if min_open_interest > 0:
            df = df[df["openInterest"] >= min_open_interest]

        return df

    # ----- risk-free rate --------------------------------------------------

    @staticmethod
    def get_risk_free_rate(default: float = 0.05) -> float:
        """Fetch the 13-Week Treasury Bill rate (^IRX) as a decimal.

        Returns ``^IRX / 100``.  Falls back to *default* on failure.
        """
        try:
            tk = yf.Ticker("^IRX")
            hist = tk.history(period="5d")
            if hist.empty:
                logger.warning(
                    "^IRX history empty — falling back to default r=%.4f", default
                )
                return default
            rate = float(hist["Close"].dropna().iloc[-1]) / 100.0
            logger.info("Risk-free rate (^IRX): %.4f", rate)
            return rate
        except Exception as exc:
            logger.warning(
                "Failed to fetch ^IRX: %s — falling back to default r=%.4f",
                exc, default,
            )
            return default

    # ----- dividend yield --------------------------------------------------

    @staticmethod
    def get_dividend_yield(ticker: str, default: float = 0.0) -> float:
        """Fetch the trailing annual dividend yield as a decimal.

        Uses yfinance's ``trailingAnnualDividendYield``.
        Falls back to *default* if unavailable.
        """
        try:
            tk = yf.Ticker(ticker)
            info = tk.info
            q = info.get("trailingAnnualDividendYield")
            if q is None or not isinstance(q, (int, float)) or q < 0:
                logger.info(
                    "Dividend yield unavailable for '%s' — using default q=%.4f",
                    ticker, default,
                )
                return default
            logger.info("Dividend yield for '%s': %.4f", ticker, float(q))
            return float(q)
        except Exception as exc:
            logger.warning(
                "Failed to fetch dividend yield for '%s': %s — using q=%.4f",
                ticker, exc, default,
            )
            return default
