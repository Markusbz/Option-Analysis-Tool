"""
Unit tests for the Black-Scholes-Merton engine.

All expected values are hardcoded from analytical computation — NOT from
the functions under test.  Primary reference case is Hull's textbook:

    S=49, K=50, r=0.05, T=20/52≈0.3846, σ=0.20, q=0
    Expected call ≈ 2.4005, Expected put ≈ 2.4482

Reference: John C. Hull, *Options, Futures, and Other Derivatives*, 11th Ed.
"""

import numpy as np
import pytest
from scipy.stats import norm

from engine import black_scholes as bsm


# ===================================================================
# Fixtures — Hull baseline inputs
# ===================================================================

HULL_S = 49.0
HULL_K = 50.0
HULL_R = 0.05
HULL_T = 20.0 / 52.0  # 20 weeks in years ≈ 0.384615
HULL_SIGMA = 0.20
HULL_Q = 0.0

# Pre-computed reference values (see module docstring)
HULL_D1 = 0.054181
HULL_D2 = -0.069853
HULL_CALL = 2.4005
HULL_PUT = 2.4482
HULL_CALL_DELTA = 0.5216
HULL_PUT_DELTA = -0.4784
HULL_GAMMA = 0.065544
HULL_VEGA = 12.1055
HULL_CALL_THETA = -4.3053   # per year
HULL_PUT_THETA = -1.8529    # per year
HULL_CALL_RHO = 8.9070
HULL_PUT_RHO = -9.9575


# ===================================================================
# d1 / d2
# ===================================================================

class TestD1D2:
    def test_d1_hull(self):
        result = float(bsm.d1(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, HULL_Q))
        assert result == pytest.approx(HULL_D1, abs=1e-4)

    def test_d2_hull(self):
        result = float(bsm.d2(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, HULL_Q))
        assert result == pytest.approx(HULL_D2, abs=1e-4)

    def test_d2_equals_d1_minus_vol_sqrt_t(self):
        """Verify the algebraic identity d2 = d1 - σ√T."""
        _d1 = bsm.d1(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, HULL_Q)
        _d2 = bsm.d2(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, HULL_Q)
        expected_d2 = _d1 - HULL_SIGMA * np.sqrt(HULL_T)
        assert float(_d2) == pytest.approx(float(expected_d2), abs=1e-10)


# ===================================================================
# Pricing
# ===================================================================

class TestPricing:
    def test_call_price_hull(self):
        result = float(bsm.call_price(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, HULL_Q))
        assert result == pytest.approx(HULL_CALL, abs=0.01)

    def test_put_price_hull(self):
        result = float(bsm.put_price(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, HULL_Q))
        assert result == pytest.approx(HULL_PUT, abs=0.01)

    def test_put_call_parity(self):
        """C − P = S·e^{−qT} − K·e^{−rT}  (must hold exactly)."""
        C = float(bsm.call_price(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, HULL_Q))
        P = float(bsm.put_price(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, HULL_Q))
        lhs = C - P
        rhs = (HULL_S * np.exp(-HULL_Q * HULL_T)
               - HULL_K * np.exp(-HULL_R * HULL_T))
        assert lhs == pytest.approx(rhs, abs=1e-8)

    def test_option_price_dispatch(self):
        """option_price() delegates correctly to call_price / put_price."""
        c1 = bsm.call_price(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA)
        c2 = bsm.option_price(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, option_type="call")
        assert float(c1) == pytest.approx(float(c2), abs=1e-12)

        p1 = bsm.put_price(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA)
        p2 = bsm.option_price(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, option_type="put")
        assert float(p1) == pytest.approx(float(p2), abs=1e-12)

    def test_invalid_option_type(self):
        with pytest.raises(ValueError, match="option_type"):
            bsm.option_price(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, option_type="straddle")

    def test_deep_itm_call(self):
        """Deep ITM call ≈ S·e^{-qT} − K·e^{-rT}  (intrinsic)."""
        S, K = 150.0, 50.0
        result = float(bsm.call_price(S, K, 0.5, 0.05, 0.20))
        intrinsic = S - K * np.exp(-0.05 * 0.5)
        assert result == pytest.approx(intrinsic, rel=0.01)

    def test_deep_otm_call_near_zero(self):
        """Deep OTM call should be near zero."""
        result = float(bsm.call_price(10.0, 100.0, 0.1, 0.05, 0.20))
        assert result == pytest.approx(0.0, abs=1e-6)


# ===================================================================
# Greeks — Delta
# ===================================================================

class TestDelta:
    def test_call_delta_hull(self):
        result = float(bsm.delta(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, HULL_Q, "call"))
        assert result == pytest.approx(HULL_CALL_DELTA, abs=0.001)

    def test_put_delta_hull(self):
        result = float(bsm.delta(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, HULL_Q, "put"))
        assert result == pytest.approx(HULL_PUT_DELTA, abs=0.001)

    def test_call_delta_bounded_0_1(self):
        """Call delta must be in [0, 1] for q=0."""
        spots = np.linspace(20, 200, 100)
        deltas = bsm.delta(spots, 100.0, 0.5, 0.05, 0.30, 0.0, "call")
        assert np.all(deltas >= 0.0)
        assert np.all(deltas <= 1.0)

    def test_put_delta_bounded_neg1_0(self):
        """Put delta must be in [-1, 0] for q=0."""
        spots = np.linspace(20, 200, 100)
        deltas = bsm.delta(spots, 100.0, 0.5, 0.05, 0.30, 0.0, "put")
        assert np.all(deltas >= -1.0)
        assert np.all(deltas <= 0.0)

    def test_call_put_delta_relationship(self):
        """Call Δ − Put Δ = e^{-qT}  (for same strike/expiry)."""
        call_d = bsm.delta(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, HULL_Q, "call")
        put_d = bsm.delta(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, HULL_Q, "put")
        assert float(call_d - put_d) == pytest.approx(
            np.exp(-HULL_Q * HULL_T), abs=1e-8
        )


# ===================================================================
# Greeks — Gamma
# ===================================================================

class TestGamma:
    def test_gamma_hull(self):
        result = float(bsm.gamma(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, HULL_Q))
        assert result == pytest.approx(HULL_GAMMA, abs=0.001)

    def test_gamma_always_positive(self):
        spots = np.linspace(30, 70, 50)
        gammas = bsm.gamma(spots, HULL_K, HULL_T, HULL_R, HULL_SIGMA, HULL_Q)
        assert np.all(gammas > 0)

    def test_gamma_peaks_near_atm(self):
        """Gamma should be highest near ATM."""
        spots = np.linspace(30, 70, 200)
        gammas = bsm.gamma(spots, HULL_K, HULL_T, HULL_R, HULL_SIGMA, HULL_Q)
        peak_idx = np.argmax(gammas)
        peak_spot = spots[peak_idx]
        # Peak should be within ±5 of the strike
        assert abs(peak_spot - HULL_K) < 5.0


# ===================================================================
# Greeks — Vega
# ===================================================================

class TestVega:
    def test_vega_hull(self):
        result = float(bsm.vega(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, HULL_Q))
        assert result == pytest.approx(HULL_VEGA, abs=0.01)

    def test_vega_always_positive(self):
        spots = np.linspace(30, 70, 50)
        vegas = bsm.vega(spots, HULL_K, HULL_T, HULL_R, HULL_SIGMA, HULL_Q)
        assert np.all(vegas > 0)


# ===================================================================
# Greeks — Theta
# ===================================================================

class TestTheta:
    def test_call_theta_hull(self):
        result = float(bsm.theta(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, HULL_Q, "call"))
        assert result == pytest.approx(HULL_CALL_THETA, abs=0.01)

    def test_put_theta_hull(self):
        result = float(bsm.theta(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, HULL_Q, "put"))
        assert result == pytest.approx(HULL_PUT_THETA, abs=0.01)

    def test_call_theta_negative_atm(self):
        """ATM call theta should be negative (time decay)."""
        result = float(bsm.theta(50.0, 50.0, 0.5, 0.05, 0.20, 0.0, "call"))
        assert result < 0


# ===================================================================
# Greeks — Rho
# ===================================================================

class TestRho:
    def test_call_rho_hull(self):
        result = float(bsm.rho(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, HULL_Q, "call"))
        assert result == pytest.approx(HULL_CALL_RHO, abs=0.01)

    def test_put_rho_hull(self):
        result = float(bsm.rho(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, HULL_Q, "put"))
        assert result == pytest.approx(HULL_PUT_RHO, abs=0.01)

    def test_call_rho_positive(self):
        """Call rho should be positive (rates up → calls worth more)."""
        assert float(bsm.rho(50, 50, 0.5, 0.05, 0.20, 0.0, "call")) > 0

    def test_put_rho_negative(self):
        """Put rho should be negative."""
        assert float(bsm.rho(50, 50, 0.5, 0.05, 0.20, 0.0, "put")) < 0


# ===================================================================
# Edge cases & vectorization
# ===================================================================

class TestEdgeCases:
    def test_t_zero_no_nan(self):
        """T=0 should be clamped, not produce NaN."""
        result = bsm.call_price(50, 50, 0.0, 0.05, 0.20)
        assert np.isfinite(result)

    def test_t_negative_clamped(self):
        """Negative T should be clamped to 1e-7."""
        result = bsm.call_price(50, 50, -1.0, 0.05, 0.20)
        assert np.isfinite(result)

    def test_vectorized_spot_array(self):
        """Functions must accept arrays for S."""
        spots = np.array([40.0, 45.0, 50.0, 55.0, 60.0])
        prices = bsm.call_price(spots, 50.0, 0.5, 0.05, 0.20)
        assert prices.shape == (5,)
        # Prices should be monotonically increasing with spot
        assert np.all(np.diff(prices) > 0)

    def test_vectorized_grid_broadcast(self):
        """S (column) × T (row) grid via broadcasting."""
        spots = np.array([40, 45, 50, 55, 60]).reshape(-1, 1)   # (5,1)
        times = np.array([0.1, 0.25, 0.5, 1.0]).reshape(1, -1)  # (1,4)
        prices = bsm.call_price(spots, 50, times, 0.05, 0.20)
        assert prices.shape == (5, 4)
        assert np.all(np.isfinite(prices))

    def test_all_greeks_finite_on_grid(self):
        """All Greeks must be finite across a reasonable S×T grid."""
        spots = np.linspace(30, 70, 20).reshape(-1, 1)
        times = np.linspace(0.01, 1.0, 10).reshape(1, -1)
        for func_name in ("delta", "gamma", "theta", "vega", "rho"):
            func = getattr(bsm, func_name)
            if func_name in ("gamma", "vega"):
                result = func(spots, 50, times, 0.05, 0.20, 0.0)
            else:
                result = func(spots, 50, times, 0.05, 0.20, 0.0, "call")
            assert np.all(np.isfinite(result)), f"{func_name} produced non-finite values"


# ===================================================================
# Dividend yield integration
# ===================================================================

class TestDividendYield:
    def test_dividend_reduces_call_price(self):
        """A positive dividend yield should reduce call price."""
        c_no_div = float(bsm.call_price(50, 50, 0.5, 0.05, 0.20, q=0.0))
        c_with_div = float(bsm.call_price(50, 50, 0.5, 0.05, 0.20, q=0.03))
        assert c_with_div < c_no_div

    def test_dividend_increases_put_price(self):
        """A positive dividend yield should increase put price."""
        p_no_div = float(bsm.put_price(50, 50, 0.5, 0.05, 0.20, q=0.0))
        p_with_div = float(bsm.put_price(50, 50, 0.5, 0.05, 0.20, q=0.03))
        assert p_with_div > p_no_div

    def test_put_call_parity_with_dividend(self):
        """Put-call parity must hold with dividends: C-P = Se^{-qT} - Ke^{-rT}."""
        S, K, T, r, sigma, q = 100, 100, 1.0, 0.05, 0.25, 0.02
        C = float(bsm.call_price(S, K, T, r, sigma, q))
        P = float(bsm.put_price(S, K, T, r, sigma, q))
        expected = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert C - P == pytest.approx(expected, abs=1e-8)


# ===================================================================
# Black-76 model
# ===================================================================

class TestBlack76:
    """Black-76: options on futures.  q is forced to r internally."""

    def test_black76_call_price(self):
        """Black-76 call = e^{-rT}·[F·N(d1) - K·N(d2)] with d1 using σ only.

        For Black-76 with q=r the BSM formula simplifies:
          C = e^{-rT}·[F·N(d1) - K·N(d2)]
        where d1 = [ln(F/K) + σ²T/2] / (σ√T).

        We verify against a manually computed value.
        """
        F, K, T, r, sigma = 100.0, 95.0, 0.5, 0.05, 0.25
        # Manual Black-76:
        sqrt_T = np.sqrt(T)
        _d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
        _d2 = _d1 - sigma * sqrt_T
        expected = np.exp(-r * T) * (F * norm.cdf(_d1) - K * norm.cdf(_d2))

        result = float(bsm.call_price(F, K, T, r, sigma, model="black76"))
        assert result == pytest.approx(expected, abs=1e-8)

    def test_black76_put_call_parity(self):
        """Black-76 parity: C - P = e^{-rT}·(F - K)."""
        F, K, T, r, sigma = 5000.0, 5100.0, 0.25, 0.04, 0.18
        C = float(bsm.call_price(F, K, T, r, sigma, model="black76"))
        P = float(bsm.put_price(F, K, T, r, sigma, model="black76"))
        expected = np.exp(-r * T) * (F - K)
        assert C - P == pytest.approx(expected, abs=1e-8)

    def test_black76_ignores_q_parameter(self):
        """When model='black76', the q parameter should be overridden."""
        F, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        # These should be identical regardless of q passed
        c1 = float(bsm.call_price(F, K, T, r, sigma, q=0.0, model="black76"))
        c2 = float(bsm.call_price(F, K, T, r, sigma, q=0.10, model="black76"))
        assert c1 == pytest.approx(c2, abs=1e-12)

    def test_black76_delta(self):
        """Black-76 call delta = e^{-rT}·N(d1)."""
        F, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.20
        sqrt_T = np.sqrt(T)
        _d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
        expected = np.exp(-r * T) * norm.cdf(_d1)

        result = float(bsm.delta(F, K, T, r, sigma, model="black76"))
        assert result == pytest.approx(expected, abs=1e-8)

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError, match="model"):
            bsm.call_price(100, 100, 0.5, 0.05, 0.20, model="binomial")
