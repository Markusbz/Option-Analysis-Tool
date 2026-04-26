"""
Strategy builder for multi-leg option portfolios.

Provides an ``OptionLeg`` dataclass and a ``Strategy`` aggregator that
computes portfolio-level PnL and Greeks by summing across legs, respecting
direction (long +1 / short −1), quantity, and the standard contract
multiplier of 100 shares per contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Optional

import numpy as np

from engine import black_scholes as bsm


# Standard equity option multiplier (shares per contract)
CONTRACT_MULTIPLIER: int = 100

# Industry-standard display scaling for Greeks.
# The BSM engine returns raw annualised / per-unit values.
# We rescale so the UI matches trading conventions:
#   Theta → daily decay  (÷365)
#   Vega  → per 1% IV     (÷100)
#   Rho   → per 1% rate   (÷100)
GREEK_DISPLAY_SCALE: dict[str, float] = {
    "delta": 1.0,
    "gamma": 1.0,
    "theta": 1.0 / 365.0,
    "vega":  1.0 / 100.0,
    "rho":   1.0 / 100.0,
}


@dataclass
class OptionLeg:
    """A single option leg in a strategy.

    Attributes
    ----------
    option_type : str
        ``'call'`` or ``'put'``.
    direction : str
        ``'long'`` (+1) or ``'short'`` (−1).
    strike : float
        Strike price.
    expiry_date : str
        ISO-format date string (e.g. ``'2026-06-20'``).
    quantity : int
        Number of contracts (always positive).
    iv : float
        Implied volatility as decimal (e.g. 0.20 for 20 %).
    premium : float
        Entry premium per share (mid price at trade time).
    """

    option_type: str
    direction: str
    strike: float
    expiry_date: str
    quantity: int = 1
    iv: float = 0.20
    premium: float = 0.0
    model: str = "bsm"  # 'bsm' or 'black76'

    # --- derived -----------------------------------------------------------

    @property
    def sign(self) -> int:
        """Return +1 for long, −1 for short."""
        return 1 if self.direction == "long" else -1

    @property
    def net_quantity(self) -> int:
        """Signed quantity (positive for long, negative for short)."""
        return self.sign * self.quantity

    # --- per-leg computations (single-contract, per-share) -----------------

    def theoretical_price(self, S, T, r, q=0.0, iv_override=None):
        """BSM / Black-76 theoretical price per share."""
        if T <= 0.0:
            # Intrinsic value at expiration
            S_arr = np.asarray(S, dtype=np.float64)
            if self.option_type == "call":
                return np.maximum(S_arr - self.strike, 0.0)
            elif self.option_type == "put":
                return np.maximum(self.strike - S_arr, 0.0)

        sigma = iv_override if iv_override is not None else self.iv
        return bsm.option_price(
            S, self.strike, T, r, sigma, q, self.option_type, model=self.model
        )

    def pnl_per_share(self, S, T, r, q=0.0, iv_override=None):
        """Unrealised PnL per share for this leg.

        PnL = direction × (current_price − entry_premium)
        """
        current = self.theoretical_price(S, T, r, q, iv_override)
        return self.sign * (current - self.premium)

    def greek(self, greek_name, S, T, r, q=0.0, iv_override=None):
        """Compute a single Greek for this leg (per share, unsigned).

        ``greek_name`` must be one of:
        ``'delta'``, ``'gamma'``, ``'theta'``, ``'vega'``, ``'rho'``.
        """
        if T <= 0.0:
            S_arr = np.asarray(S, dtype=np.float64)
            if greek_name == "delta":
                if self.option_type == "call":
                    return np.where(S_arr > self.strike, 1.0, 0.0)
                elif self.option_type == "put":
                    return np.where(S_arr < self.strike, -1.0, 0.0)
            return np.zeros_like(S_arr)

        sigma = iv_override if iv_override is not None else self.iv
        func = _GREEK_FUNCS[greek_name]
        # gamma and vega don't take option_type
        if greek_name in ("gamma", "vega"):
            return func(S, self.strike, T, r, sigma, q, model=self.model)
        return func(S, self.strike, T, r, sigma, q, self.option_type, model=self.model)


# Map Greek names → BSM functions
_GREEK_FUNCS = {
    "delta": bsm.delta,
    "gamma": bsm.gamma,
    "theta": bsm.theta,
    "vega": bsm.vega,
    "rho": bsm.rho,
}

GREEK_NAMES = list(_GREEK_FUNCS.keys())


class Strategy:
    """Multi-leg option strategy with portfolio-level aggregation.

    All aggregated values respect:
    - ``direction``  (long +1 / short −1)
    - ``quantity``   (number of contracts)
    - ``CONTRACT_MULTIPLIER`` (100 shares per contract)

    Parameters
    ----------
    spot : float
        Current underlying price (used for template generation).
    risk_free_rate : float
        Annualised risk-free rate (decimal).
    dividend_yield : float
        Continuous dividend yield (decimal).
    """

    def __init__(
        self,
        spot: float = 0.0,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0,
    ):
        self.spot = spot
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.legs: List[OptionLeg] = []

    # --- leg management ----------------------------------------------------

    def add_leg(self, leg: OptionLeg) -> None:
        self.legs.append(leg)

    def remove_leg(self, index: int) -> None:
        if 0 <= index < len(self.legs):
            self.legs.pop(index)

    def clear_legs(self) -> None:
        self.legs.clear()

    # --- per-leg time-to-expiry --------------------------------------------

    @staticmethod
    def _leg_T(leg: OptionLeg, dte_offset_days: float | None = None) -> float:
        """Compute annualised T for *this* leg from its expiry_date.

        Parameters
        ----------
        leg : OptionLeg
            The option leg with an ``expiry_date`` string.
        dte_offset_days : float | None
            If given, shift (reduce) the per-leg DTE by this many days.
            Useful for the "What-If" DTE slider: the slider says "show me
            this strategy as if N fewer calendar days remain".
            When ``None``, no shift is applied (compute from today).

        Returns
        -------
        float
            T in years, clamped to >= 1e-7.
        """
        try:
            exp = datetime.strptime(leg.expiry_date, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            # Unparseable → fall back to 30 days
            return 30.0 / 365.0

        dte = (exp - date.today()).days
        if dte_offset_days is not None:
            dte -= dte_offset_days
        return dte / 365.0

    # --- portfolio-level computations --------------------------------------

    def total_pnl(
        self,
        spot_range: np.ndarray,
        T: float | None = None,
        iv_shift: float = 0.0,
        *,
        dte_offset_days: float | None = None,
    ) -> np.ndarray:
        """Aggregate PnL in dollar terms across all legs.

        Parameters
        ----------
        spot_range : array-like
            Underlying prices to evaluate.
        T : float | None
            Global blanket T (legacy).  If *dte_offset_days* is given,
            each leg computes its own T from its ``expiry_date``.
        iv_shift : float
            Additive shift to each leg's IV (e.g. +0.05 = +5 pp).
        dte_offset_days : float | None
            Per-leg DTE shift (days to subtract from each leg's DTE).

        Returns
        -------
        np.ndarray
            Total PnL for the portfolio at each spot price.
        """
        spot_range = np.asarray(spot_range, dtype=np.float64)
        total = np.zeros_like(spot_range)
        for leg in self.legs:
            iv = max(leg.iv + iv_shift, 0.001)  # clamp: IV can never go negative
            leg_t = self._resolve_T(leg, T, dte_offset_days)
            pnl = leg.pnl_per_share(
                spot_range, leg_t, self.risk_free_rate, self.dividend_yield, iv
            )
            # Scale by quantity, direction is already in pnl_per_share
            total += pnl * leg.quantity * CONTRACT_MULTIPLIER
        return total

    def total_greek(
        self,
        greek_name: str,
        spot_range: np.ndarray,
        T: float | None = None,
        iv_shift: float = 0.0,
        *,
        dte_offset_days: float | None = None,
    ) -> np.ndarray:
        """Aggregate a Greek across all legs.

        The result is scaled by direction × quantity × CONTRACT_MULTIPLIER,
        then by GREEK_DISPLAY_SCALE for industry-standard units.
        """
        spot_range = np.asarray(spot_range, dtype=np.float64)
        total = np.zeros_like(spot_range)
        for leg in self.legs:
            iv = max(leg.iv + iv_shift, 0.001)  # clamp: IV can never go negative
            leg_t = self._resolve_T(leg, T, dte_offset_days)
            g = leg.greek(
                greek_name, spot_range, leg_t,
                self.risk_free_rate, self.dividend_yield, iv,
            )
            total += leg.sign * leg.quantity * CONTRACT_MULTIPLIER * g
        # Apply industry-standard display scaling
        total *= GREEK_DISPLAY_SCALE.get(greek_name, 1.0)
        return total

    def _resolve_T(
        self,
        leg: OptionLeg,
        T_global: float | None,
        dte_offset_days: float | None,
    ) -> float:
        """Decide which T to use for a given leg.

        If dte_offset_days is provided → per-leg T from expiry_date.
        Otherwise → fall back to T_global (legacy path).
        """
        if dte_offset_days is not None:
            return self._leg_T(leg, dte_offset_days)
        if T_global is not None:
            return T_global
        # Absolute fallback: per-leg from expiry_date, no offset
        return self._leg_T(leg, None)

    def nearest_dte(self) -> float:
        """Return the actual DTE of the nearest-term active leg."""
        if not self.legs:
            return 0.0
        
        dtes = []
        for leg in self.legs:
            try:
                exp = datetime.strptime(leg.expiry_date, "%Y-%m-%d").date()
                dtes.append(float((exp - date.today()).days))
            except (ValueError, TypeError):
                dtes.append(30.0)
        return min(dtes)

    def net_greeks_at_spot(
        self,
        T: float,
        iv_shift: float = 0.0,
    ) -> dict:
        """Return a dict of net Greeks evaluated at current ``self.spot``."""
        S = np.array([self.spot])
        return {
            name: float(self.total_greek(name, S, T, iv_shift)[0])
            for name in GREEK_NAMES
        }

    # --- pre-built templates -----------------------------------------------

    @classmethod
    def from_template(
        cls,
        name: str,
        spot: float,
        strikes: List[float],
        expiry: str,
        ivs: List[float],
        premiums: List[float],
        r: float = 0.05,
        q: float = 0.0,
        model: str = "bsm",
    ) -> "Strategy":
        """Create a Strategy from a named template.

        Supported templates
        -------------------
        - ``'long_call'``         — 1 strike
        - ``'long_put'``          — 1 strike
        - ``'bull_call_spread'``  — 2 strikes [lower, upper]
        - ``'bear_put_spread'``   — 2 strikes [upper, lower]  (buy high, sell low)
        - ``'long_straddle'``     — 1 strike (ATM)
        - ``'long_strangle'``     — 2 strikes [lower_put, upper_call]
        - ``'iron_condor'``       — 4 strikes [put_buy, put_sell, call_sell, call_buy]
        - ``'butterfly'``         — 3 strikes [lower, middle, upper]
        """
        strat = cls(spot=spot, risk_free_rate=r, dividend_yield=q)
        templates = _TEMPLATES[name]
        for i, (opt_type, direction) in enumerate(templates):
            strat.add_leg(OptionLeg(
                option_type=opt_type,
                direction=direction,
                strike=strikes[i],
                expiry_date=expiry,
                quantity=2 if name == "butterfly" and i == 1 else 1,
                iv=ivs[i] if i < len(ivs) else ivs[-1],
                premium=premiums[i] if i < len(premiums) else premiums[-1],
                model=model,
            ))
        return strat


# Template definitions: list of (option_type, direction) per leg
_TEMPLATES = {
    "long_call": [("call", "long")],
    "long_put": [("put", "long")],
    "bull_call_spread": [("call", "long"), ("call", "short")],
    "bear_put_spread": [("put", "long"), ("put", "short")],
    "long_straddle": [("call", "long"), ("put", "long")],
    "long_strangle": [("put", "long"), ("call", "long")],
    "iron_condor": [
        ("put", "long"),    # OTM put protection (wing)
        ("put", "short"),   # OTM put sold
        ("call", "short"),  # OTM call sold
        ("call", "long"),   # OTM call protection (wing)
    ],
    "butterfly": [
        ("call", "long"),   # lower strike
        ("call", "short"),  # middle strike (qty=2)
        ("call", "long"),   # upper strike
    ],
}
