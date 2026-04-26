"""
Unit tests for the Strategy aggregator.

Tests verify:
- Single-leg PnL and Greek computation
- Direction handling (long vs short)
- Quantity and CONTRACT_MULTIPLIER scaling
- Multi-leg aggregation (bull call spread, iron condor)
- Pre-built template construction

Expected values are computed manually or via known analytical results,
NOT by calling the functions under test.
"""

import numpy as np
import pytest

from engine.strategy import OptionLeg, Strategy, CONTRACT_MULTIPLIER
from engine import black_scholes as bsm


# ===================================================================
# OptionLeg basics
# ===================================================================

class TestOptionLeg:
    def test_sign_long(self):
        leg = OptionLeg("call", "long", 100.0, "2026-06-20")
        assert leg.sign == 1

    def test_sign_short(self):
        leg = OptionLeg("call", "short", 100.0, "2026-06-20")
        assert leg.sign == -1

    def test_net_quantity_long(self):
        leg = OptionLeg("call", "long", 100.0, "2026-06-20", quantity=3)
        assert leg.net_quantity == 3

    def test_net_quantity_short(self):
        leg = OptionLeg("put", "short", 100.0, "2026-06-20", quantity=5)
        assert leg.net_quantity == -5


# ===================================================================
# Single-leg PnL
# ===================================================================

class TestSingleLegPnL:
    """Verify PnL for a single option leg at expiration (T≈0)."""

    def test_long_call_itm_at_expiry(self):
        """Long call at expiry, ITM: PnL = (S - K) - premium."""
        leg = OptionLeg("call", "long", 100.0, "2026-06-20",
                        quantity=1, iv=0.20, premium=5.0)
        # At expiry (T≈0), call price = max(S-K, 0) = 10
        S = np.array([110.0])
        pnl = leg.pnl_per_share(S, T=1e-7, r=0.05, q=0.0)
        # PnL = +1 × (10 - 5) = 5.0 per share
        assert float(pnl[0]) == pytest.approx(5.0, abs=0.01)

    def test_long_call_otm_at_expiry(self):
        """Long call at expiry, OTM: PnL = -premium."""
        leg = OptionLeg("call", "long", 100.0, "2026-06-20",
                        quantity=1, iv=0.20, premium=5.0)
        S = np.array([90.0])
        pnl = leg.pnl_per_share(S, T=1e-7, r=0.05, q=0.0)
        # PnL = +1 × (0 - 5) = -5.0
        assert float(pnl[0]) == pytest.approx(-5.0, abs=0.01)

    def test_short_put_itm_at_expiry(self):
        """Short put at expiry, ITM: PnL = -(K-S) + premium."""
        leg = OptionLeg("put", "short", 100.0, "2026-06-20",
                        quantity=1, iv=0.20, premium=4.0)
        S = np.array([90.0])
        pnl = leg.pnl_per_share(S, T=1e-7, r=0.05, q=0.0)
        # put_price ≈ max(100-90, 0) = 10
        # PnL = -1 × (10 - 4) = -6.0
        assert float(pnl[0]) == pytest.approx(-6.0, abs=0.01)


# ===================================================================
# Strategy — direction & quantity scaling
# ===================================================================

class TestStrategyScaling:
    def test_long_short_cancel(self):
        """Identical long + short legs should net to ~zero Greek."""
        strat = Strategy(spot=100.0, risk_free_rate=0.05, dividend_yield=0.0)
        strat.add_leg(OptionLeg("call", "long", 100.0, "2026-06-20",
                                quantity=1, iv=0.20, premium=5.0))
        strat.add_leg(OptionLeg("call", "short", 100.0, "2026-06-20",
                                quantity=1, iv=0.20, premium=5.0))

        S = np.array([100.0])
        net_delta = strat.total_greek("delta", S, T=0.5)
        net_pnl = strat.total_pnl(S, T=0.5)
        assert float(net_delta[0]) == pytest.approx(0.0, abs=1e-8)
        assert float(net_pnl[0]) == pytest.approx(0.0, abs=1e-8)

    def test_quantity_multiplier(self):
        """Buying 3 contracts should give 3× the single-contract Greek."""
        strat_1 = Strategy(spot=100.0, risk_free_rate=0.05)
        strat_1.add_leg(OptionLeg("call", "long", 100.0, "2026-06-20",
                                  quantity=1, iv=0.20))

        strat_3 = Strategy(spot=100.0, risk_free_rate=0.05)
        strat_3.add_leg(OptionLeg("call", "long", 100.0, "2026-06-20",
                                  quantity=3, iv=0.20))

        S = np.array([100.0])
        d1 = strat_1.total_greek("delta", S, T=0.5)
        d3 = strat_3.total_greek("delta", S, T=0.5)
        assert float(d3[0]) == pytest.approx(3.0 * float(d1[0]), abs=1e-8)

    def test_contract_multiplier_applied(self):
        """Total Greek should include the 100-shares-per-contract multiplier."""
        strat = Strategy(spot=50.0, risk_free_rate=0.05)
        strat.add_leg(OptionLeg("call", "long", 50.0, "2026-06-20",
                                quantity=1, iv=0.20))

        S = np.array([50.0])
        # Raw BSM delta for 1 share
        raw_delta = float(bsm.delta(50.0, 50.0, 0.5, 0.05, 0.20, 0.0, "call"))
        strat_delta = float(strat.total_greek("delta", S, T=0.5)[0])

        # Strategy delta = raw_delta × 1 (sign) × 1 (qty) × 100 (multiplier)
        assert strat_delta == pytest.approx(raw_delta * CONTRACT_MULTIPLIER, abs=1e-6)


# ===================================================================
# Strategy — multi-leg aggregation
# ===================================================================

class TestMultiLegStrategies:
    def test_bull_call_spread_max_loss(self):
        """Bull call spread: max loss = net debit (at S << lower strike)."""
        lower_K, upper_K = 95.0, 105.0
        premium_long, premium_short = 8.0, 3.0
        net_debit = premium_long - premium_short  # 5.0

        strat = Strategy(spot=100.0, risk_free_rate=0.05)
        strat.add_leg(OptionLeg("call", "long", lower_K, "2026-06-20",
                                quantity=1, iv=0.20, premium=premium_long))
        strat.add_leg(OptionLeg("call", "short", upper_K, "2026-06-20",
                                quantity=1, iv=0.20, premium=premium_short))

        # At expiry, with S well below both strikes → both expire worthless
        S_low = np.array([80.0])
        pnl = strat.total_pnl(S_low, T=1e-7)
        expected_loss = -net_debit * CONTRACT_MULTIPLIER
        assert float(pnl[0]) == pytest.approx(expected_loss, abs=1.0)

    def test_bull_call_spread_max_profit(self):
        """Bull call spread: max profit = (spread width - net debit) at S >> upper strike."""
        lower_K, upper_K = 95.0, 105.0
        premium_long, premium_short = 8.0, 3.0
        net_debit = premium_long - premium_short
        max_profit = (upper_K - lower_K) - net_debit  # 10 - 5 = 5

        strat = Strategy(spot=100.0, risk_free_rate=0.05)
        strat.add_leg(OptionLeg("call", "long", lower_K, "2026-06-20",
                                quantity=1, iv=0.20, premium=premium_long))
        strat.add_leg(OptionLeg("call", "short", upper_K, "2026-06-20",
                                quantity=1, iv=0.20, premium=premium_short))

        S_high = np.array([150.0])
        pnl = strat.total_pnl(S_high, T=1e-7)
        expected_profit = max_profit * CONTRACT_MULTIPLIER
        assert float(pnl[0]) == pytest.approx(expected_profit, abs=1.0)

    def test_straddle_delta_near_zero_atm(self):
        """ATM straddle should have near-zero net delta."""
        strat = Strategy(spot=100.0, risk_free_rate=0.05)
        strat.add_leg(OptionLeg("call", "long", 100.0, "2026-06-20",
                                quantity=1, iv=0.20))
        strat.add_leg(OptionLeg("put", "long", 100.0, "2026-06-20",
                                quantity=1, iv=0.20))

        S = np.array([100.0])
        net_delta = strat.total_greek("delta", S, T=0.5)
        # Call delta ≈ 0.56, put delta ≈ -0.44 → net ≈ 0.12 × 100
        # (not exactly zero due to forward pricing, but should be small)
        assert abs(float(net_delta[0])) < 20.0  # < 0.20 per share × 100

    def test_straddle_positive_gamma(self):
        """Long straddle should have positive gamma (long vol)."""
        strat = Strategy(spot=100.0, risk_free_rate=0.05)
        strat.add_leg(OptionLeg("call", "long", 100.0, "2026-06-20",
                                quantity=1, iv=0.20))
        strat.add_leg(OptionLeg("put", "long", 100.0, "2026-06-20",
                                quantity=1, iv=0.20))

        S = np.array([100.0])
        net_gamma = strat.total_greek("gamma", S, T=0.5)
        assert float(net_gamma[0]) > 0


# ===================================================================
# Strategy — IV shift
# ===================================================================

class TestIVShift:
    def test_iv_shift_increases_option_value(self):
        """Positive IV shift should increase the value of a long straddle."""
        strat = Strategy(spot=100.0, risk_free_rate=0.05)
        strat.add_leg(OptionLeg("call", "long", 100.0, "2026-06-20",
                                quantity=1, iv=0.20, premium=5.0))
        strat.add_leg(OptionLeg("put", "long", 100.0, "2026-06-20",
                                quantity=1, iv=0.20, premium=5.0))

        S = np.array([100.0])
        pnl_base = float(strat.total_pnl(S, T=0.5, iv_shift=0.0)[0])
        pnl_up = float(strat.total_pnl(S, T=0.5, iv_shift=0.10)[0])
        assert pnl_up > pnl_base


# ===================================================================
# Strategy — leg management
# ===================================================================

class TestLegManagement:
    def test_add_remove_legs(self):
        strat = Strategy(spot=100.0)
        strat.add_leg(OptionLeg("call", "long", 100.0, "2026-06-20"))
        strat.add_leg(OptionLeg("put", "long", 100.0, "2026-06-20"))
        assert len(strat.legs) == 2

        strat.remove_leg(0)
        assert len(strat.legs) == 1
        assert strat.legs[0].option_type == "put"

    def test_clear_legs(self):
        strat = Strategy(spot=100.0)
        strat.add_leg(OptionLeg("call", "long", 100.0, "2026-06-20"))
        strat.clear_legs()
        assert len(strat.legs) == 0

    def test_remove_invalid_index(self):
        """Removing out-of-range index should be a no-op."""
        strat = Strategy(spot=100.0)
        strat.add_leg(OptionLeg("call", "long", 100.0, "2026-06-20"))
        strat.remove_leg(5)  # out of range
        assert len(strat.legs) == 1


# ===================================================================
# Strategy — templates
# ===================================================================

class TestTemplates:
    def test_bull_call_spread_template(self):
        strat = Strategy.from_template(
            "bull_call_spread", spot=100.0,
            strikes=[95.0, 105.0], expiry="2026-06-20",
            ivs=[0.20, 0.22], premiums=[8.0, 3.0],
        )
        assert len(strat.legs) == 2
        assert strat.legs[0].option_type == "call"
        assert strat.legs[0].direction == "long"
        assert strat.legs[0].strike == 95.0
        assert strat.legs[1].option_type == "call"
        assert strat.legs[1].direction == "short"
        assert strat.legs[1].strike == 105.0

    def test_iron_condor_template(self):
        strat = Strategy.from_template(
            "iron_condor", spot=100.0,
            strikes=[85.0, 90.0, 110.0, 115.0],
            expiry="2026-06-20",
            ivs=[0.25, 0.22, 0.22, 0.25],
            premiums=[1.0, 2.5, 2.5, 1.0],
        )
        assert len(strat.legs) == 4
        # Wings are long, body is short
        assert strat.legs[0].direction == "long"   # put buy
        assert strat.legs[1].direction == "short"  # put sell
        assert strat.legs[2].direction == "short"  # call sell
        assert strat.legs[3].direction == "long"   # call buy

    def test_butterfly_middle_leg_qty_2(self):
        strat = Strategy.from_template(
            "butterfly", spot=100.0,
            strikes=[95.0, 100.0, 105.0],
            expiry="2026-06-20",
            ivs=[0.20, 0.20, 0.20],
            premiums=[8.0, 5.0, 3.0],
        )
        assert len(strat.legs) == 3
        assert strat.legs[0].quantity == 1   # lower wing
        assert strat.legs[1].quantity == 2   # middle body
        assert strat.legs[2].quantity == 1   # upper wing


# ===================================================================
# Strategy — net_greeks_at_spot
# ===================================================================

class TestNetGreeks:
    def test_net_greeks_returns_all_five(self):
        strat = Strategy(spot=100.0, risk_free_rate=0.05)
        strat.add_leg(OptionLeg("call", "long", 100.0, "2026-06-20",
                                quantity=1, iv=0.20))
        greeks = strat.net_greeks_at_spot(T=0.5)
        assert set(greeks.keys()) == {"delta", "gamma", "theta", "vega", "rho"}
        for v in greeks.values():
            assert np.isfinite(v)
