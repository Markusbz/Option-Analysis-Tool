"""
Black-Scholes-Merton (BSM) and Black-76 pricing for European options.

All functions are fully vectorized via numpy — they accept scalars or
np.ndarray inputs and leverage broadcasting for grid computations.

Models
------
- ``model='bsm'``     : Standard BSM.  *S* = spot price, *q* = dividend yield.
- ``model='black76'``  : Black-76 for options on futures.  *S* is interpreted as
  the **futures / forward price F**.  Internally this sets ``q = r`` so that
  the cost-of-carry term vanishes (``e^{-qT}·F = e^{-rT}·F``).

Conventions
-----------
- T : time to expiration in **years** (DTE / 365).  Clamped to max(T, 1e-7).
- sigma : annualized decimal volatility (0.20 = 20 %).
- q : continuous dividend yield (annualized decimal).
- Vega returned per unit change in sigma.  Divide by 100 for 1 % vol bump.
- Theta returned per year.  Divide by 365 for daily theta.
- Rho returned per unit change in r.  Divide by 100 for 1 % rate bump.

References
----------
John C. Hull, *Options, Futures, and Other Derivatives*, 11th Ed.
Fischer Black, "The pricing of commodity contracts", 1976.
"""

import numpy as np
from scipy.stats import norm

# Valid model identifiers
VALID_MODELS = ("bsm", "black76")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cast(S, K, T, r, sigma, q):
    """Cast all inputs to float64 arrays and clamp T > 0."""
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.maximum(np.asarray(T, dtype=np.float64), 1e-7)
    r = np.asarray(r, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return S, K, T, r, sigma, q


def _resolve_model(r, q, model):
    """Apply model semantics.  For Black-76, override q = r."""
    if model == "black76":
        return np.asarray(r, dtype=np.float64)   # q ← r
    if model == "bsm":
        return np.asarray(q, dtype=np.float64)
    raise ValueError(
        f"model must be one of {VALID_MODELS}, got '{model}'"
    )


def _d1_d2(S, K, T, r, sigma, q):
    """Compute d1 and d2.  Assumes inputs already cast & clamped."""
    sqrt_T = np.sqrt(T)
    vol_sqrt_T = sigma * sqrt_T
    _d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / vol_sqrt_T
    _d2 = _d1 - vol_sqrt_T
    return _d1, _d2


# ---------------------------------------------------------------------------
# Public: d1, d2
# ---------------------------------------------------------------------------

def d1(S, K, T, r, sigma, q=0.0, *, model="bsm"):
    """BSM / Black-76 d1 term."""
    q = _resolve_model(r, q, model)
    S, K, T, r, sigma, q = _cast(S, K, T, r, sigma, q)
    _d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return _d1


def d2(S, K, T, r, sigma, q=0.0, *, model="bsm"):
    """BSM / Black-76 d2 term."""
    q = _resolve_model(r, q, model)
    S, K, T, r, sigma, q = _cast(S, K, T, r, sigma, q)
    _, _d2 = _d1_d2(S, K, T, r, sigma, q)
    return _d2


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

def call_price(S, K, T, r, sigma, q=0.0, *, model="bsm"):
    """European call:  C = S·e^{-qT}·N(d1) - K·e^{-rT}·N(d2)."""
    q = _resolve_model(r, q, model)
    S, K, T, r, sigma, q = _cast(S, K, T, r, sigma, q)
    _d1, _d2 = _d1_d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.cdf(_d1) - K * np.exp(-r * T) * norm.cdf(_d2)


def put_price(S, K, T, r, sigma, q=0.0, *, model="bsm"):
    """European put:  P = K·e^{-rT}·N(-d2) - S·e^{-qT}·N(-d1)."""
    q = _resolve_model(r, q, model)
    S, K, T, r, sigma, q = _cast(S, K, T, r, sigma, q)
    _d1, _d2 = _d1_d2(S, K, T, r, sigma, q)
    return K * np.exp(-r * T) * norm.cdf(-_d2) - S * np.exp(-q * T) * norm.cdf(-_d1)


def option_price(S, K, T, r, sigma, q=0.0, option_type="call", *, model="bsm"):
    """European option price (dispatches to call_price / put_price)."""
    if option_type == "call":
        return call_price(S, K, T, r, sigma, q, model=model)
    if option_type == "put":
        return put_price(S, K, T, r, sigma, q, model=model)
    raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------

def delta(S, K, T, r, sigma, q=0.0, option_type="call", *, model="bsm"):
    """Δ = ∂V/∂S.

    Call:  e^{-qT}·N(d1)
    Put:  -e^{-qT}·N(-d1)
    """
    q = _resolve_model(r, q, model)
    S, K, T, r, sigma, q = _cast(S, K, T, r, sigma, q)
    _d1, _ = _d1_d2(S, K, T, r, sigma, q)
    disc = np.exp(-q * T)
    if option_type == "call":
        return disc * norm.cdf(_d1)
    if option_type == "put":
        return disc * (norm.cdf(_d1) - 1.0)
    raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


def gamma(S, K, T, r, sigma, q=0.0, *, model="bsm"):
    """Γ = ∂²V/∂S²  (identical for calls and puts).

    Γ = e^{-qT}·n(d1) / (S·σ·√T)
    """
    q = _resolve_model(r, q, model)
    S, K, T, r, sigma, q = _cast(S, K, T, r, sigma, q)
    _d1, _ = _d1_d2(S, K, T, r, sigma, q)
    sqrt_T = np.sqrt(T)
    return np.exp(-q * T) * norm.pdf(_d1) / (S * sigma * sqrt_T)


def vega(S, K, T, r, sigma, q=0.0, *, model="bsm"):
    """ν = ∂V/∂σ  (identical for calls and puts).

    ν = S·e^{-qT}·n(d1)·√T
    """
    q = _resolve_model(r, q, model)
    S, K, T, r, sigma, q = _cast(S, K, T, r, sigma, q)
    _d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(_d1) * np.sqrt(T)


def theta(S, K, T, r, sigma, q=0.0, option_type="call", *, model="bsm"):
    """Θ = ∂V/∂τ  (per year; divide by 365 for daily).

    Call: −S·e^{-qT}·n(d1)·σ/(2√T) − r·K·e^{-rT}·N(d2) + q·S·e^{-qT}·N(d1)
    Put:  −S·e^{-qT}·n(d1)·σ/(2√T) + r·K·e^{-rT}·N(−d2) − q·S·e^{-qT}·N(−d1)
    """
    q = _resolve_model(r, q, model)
    S, K, T, r, sigma, q = _cast(S, K, T, r, sigma, q)
    _d1, _d2 = _d1_d2(S, K, T, r, sigma, q)
    sqrt_T = np.sqrt(T)
    exp_qT = np.exp(-q * T)
    exp_rT = np.exp(-r * T)
    pdf_d1 = norm.pdf(_d1)

    # Diffusion (time-decay) component — always negative
    diffusion = -(S * exp_qT * pdf_d1 * sigma) / (2.0 * sqrt_T)

    if option_type == "call":
        return diffusion - r * K * exp_rT * norm.cdf(_d2) + q * S * exp_qT * norm.cdf(_d1)
    if option_type == "put":
        return diffusion + r * K * exp_rT * norm.cdf(-_d2) - q * S * exp_qT * norm.cdf(-_d1)
    raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


def rho(S, K, T, r, sigma, q=0.0, option_type="call", *, model="bsm"):
    """ρ = ∂V/∂r  (per unit change in r).

    Call:  K·T·e^{-rT}·N(d2)
    Put:  −K·T·e^{-rT}·N(−d2)
    """
    q = _resolve_model(r, q, model)
    S, K, T, r, sigma, q = _cast(S, K, T, r, sigma, q)
    _, _d2 = _d1_d2(S, K, T, r, sigma, q)
    exp_rT = np.exp(-r * T)
    if option_type == "call":
        return K * T * exp_rT * norm.cdf(_d2)
    if option_type == "put":
        return -K * T * exp_rT * norm.cdf(-_d2)
    raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
