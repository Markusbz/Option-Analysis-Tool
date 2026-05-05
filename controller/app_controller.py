"""
controller/app_controller.py

Wires the UI signals to the engine (BSM, Strategy, DataFetcher)
and updates the charts.  Handles:
- Threaded data fetching (never blocks the UI)
- Debounced Plotly surface updates (300ms after last slider move)
- Real-time pyqtgraph updates on slider drag
- Option chain click → leg addition
- IV clamping (negative IV prevented at engine level)
- Per-leg T computation for calendar spreads
- Safe thread lifecycle management (no dangling C++ pointers)
"""

from __future__ import annotations

from datetime import datetime, date
import logging

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QMessageBox

from engine.data_fetcher import MarketDataFetcher, MarketDataError
from engine.strategy import OptionLeg, Strategy, GREEK_NAMES
from ui.theme import Colors

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data-fetch worker (runs in QThread)
# ---------------------------------------------------------------------------

class _FetchWorker(QObject):
    """Runs yfinance calls off the main thread."""

    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, ticker: str):
        super().__init__()
        self._ticker = ticker

    @pyqtSlot()
    def run(self):
        try:
            fetcher = MarketDataFetcher()
            spot = fetcher.fetch_spot(self._ticker)
            expirations = fetcher.fetch_expirations(self._ticker)
            rate = fetcher.get_risk_free_rate()
            div_yield = fetcher.get_dividend_yield(self._ticker)
            self.finished.emit({
                "ticker": self._ticker,
                "spot": spot,
                "expirations": expirations,
                "rate": rate,
                "div_yield": div_yield,
            })
        except MarketDataError as exc:
            self.error.emit(str(exc))
        except Exception as exc:
            self.error.emit(f"Unexpected error: {exc}")


class _ChainFetchWorker(QObject):
    """Fetches a single option chain off the main thread."""

    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, ticker: str, expiry: str):
        super().__init__()
        self._ticker = ticker
        self._expiry = expiry

    @pyqtSlot()
    def run(self):
        try:
            chain = MarketDataFetcher.fetch_chain(self._ticker, self._expiry)
            self.finished.emit(chain)
        except MarketDataError as exc:
            self.error.emit(str(exc))
        except Exception as exc:
            self.error.emit(f"Unexpected error: {exc}")


# ---------------------------------------------------------------------------
# Thread helper — safe start / stop
# ---------------------------------------------------------------------------

def _shutdown_thread(thread: QThread | None) -> None:
    """Safely shut down a QThread, blocking until it finishes."""
    if thread is None:
        return
    if thread.isRunning():
        thread.quit()
        thread.wait(3000)  # max 3 s
    # If the thread is still alive (unlikely), force-terminate
    if thread.isRunning():
        thread.terminate()
        thread.wait(1000)


# ---------------------------------------------------------------------------
# Main controller
# ---------------------------------------------------------------------------

class AppController(QObject):
    """Connects ControlPanel + OptionChain signals → Engine → Charts."""

    def __init__(self, control_panel, option_chain, pnl_chart, greek_chart,
                 surface_chart, sensitivity_chart, summary_panel):
        super().__init__()

        self.cp = control_panel
        self.chain_widget = option_chain
        self.pnl_chart = pnl_chart
        self.greek_chart = greek_chart
        self.surface_chart = surface_chart
        self.sensitivity_chart = sensitivity_chart
        self.summary_panel = summary_panel

        # State
        self._ticker = ""
        self._spot = 0.0
        self._rate = 0.05
        self._div_yield = 0.0
        self._expirations = []
        self._chain = None
        self._strategy = Strategy()

        # Thread references — we keep strong refs so the QThread / Worker
        # objects survive until we explicitly shut them down.
        self._fetch_thread: QThread | None = None
        self._fetch_worker: _FetchWorker | None = None
        self._chain_thread: QThread | None = None
        self._chain_worker: _ChainFetchWorker | None = None

        # Debounce timer for Plotly surface (300ms)
        self._surface_timer = QTimer()
        self._surface_timer.setSingleShot(True)
        self._surface_timer.setInterval(300)
        self._surface_timer.timeout.connect(self._update_surface)

        # Connect control panel signals
        self.cp.fetch_requested.connect(self._on_fetch)
        self.cp.strategy_changed.connect(self._on_strategy_template)
        self.cp.expiry_changed.connect(self._on_expiry_changed)
        self.cp.params_changed.connect(self._on_params_changed)
        self.cp.clear_legs_requested.connect(self._on_clear_legs)
        self.cp.remove_leg_requested.connect(self._on_remove_leg)
        self.cp.legs_changed.connect(self._on_legs_edited)
        self.cp.reset_view_requested.connect(self._on_reset_view)

        # Connect option chain signals
        self.chain_widget.leg_added.connect(self._on_chain_leg_added)

    # ----- Fetch data (threaded) -------------------------------------------

    def _on_fetch(self, ticker: str):
        # Safely tear down any previous fetch thread
        _shutdown_thread(self._fetch_thread)

        self.cp.set_status("\u23f3 Fetching data...", Colors.ACCENT_BLUE)
        self.cp.fetch_btn.setEnabled(False)

        self._fetch_thread = QThread()
        self._fetch_worker = _FetchWorker(ticker)
        self._fetch_worker.moveToThread(self._fetch_thread)
        self._fetch_thread.started.connect(self._fetch_worker.run)
        self._fetch_worker.finished.connect(self._on_fetch_done)
        self._fetch_worker.error.connect(self._on_fetch_error)
        self._fetch_worker.finished.connect(self._fetch_thread.quit)
        self._fetch_worker.error.connect(self._fetch_thread.quit)
        self._fetch_thread.start()

    def _on_fetch_done(self, data: dict):
        self._ticker = data["ticker"]
        self._spot = data["spot"]
        self._rate = data["rate"]
        self._div_yield = data["div_yield"]
        self._expirations = data["expirations"]

        self.cp.set_spot_info(self._spot, self._rate, self._div_yield)
        self.cp.set_expirations(self._expirations)
        self.cp.rate_slider.reset(self._rate * 100.0)
        self.cp.set_status(
            f"\u2713 {self._ticker}  |  {len(self._expirations)} expirations",
            Colors.ACCENT_GREEN,
        )
        self.cp.fetch_btn.setEnabled(True)

        # Auto-fetch the first expiry chain
        if self._expirations:
            self._fetch_chain(self._ticker, self._expirations[0])

    def _on_fetch_error(self, msg: str):
        self.cp.set_status(f"\u2717 {msg}", Colors.ACCENT_RED)
        self.cp.fetch_btn.setEnabled(True)
        QMessageBox.warning(self.main_window, "Market Data Error", msg)

    # ----- Chain fetch (threaded) ------------------------------------------

    def _fetch_chain(self, ticker: str, expiry: str):
        # Safely tear down any previous chain thread
        _shutdown_thread(self._chain_thread)

        self.cp.set_status(f"\u23f3 Loading chain {expiry}...", Colors.ACCENT_BLUE)

        self._chain_thread = QThread()
        self._chain_worker = _ChainFetchWorker(ticker, expiry)
        self._chain_worker.moveToThread(self._chain_thread)
        self._chain_thread.started.connect(self._chain_worker.run)
        self._chain_worker.finished.connect(self._on_chain_done)
        self._chain_worker.error.connect(self._on_chain_error)
        self._chain_worker.finished.connect(self._chain_thread.quit)
        self._chain_worker.error.connect(self._chain_thread.quit)
        self._chain_thread.start()

    def _on_chain_done(self, chain):
        self._chain = chain
        n_calls = len(chain[chain["option_type"] == "call"])
        n_puts = len(chain[chain["option_type"] == "put"])
        self.cp.set_status(
            f"\u2713 Chain loaded  |  {n_calls}C / {n_puts}P strikes",
            Colors.ACCENT_GREEN,
        )

        # Compute actual DTE for this expiry and sync to the slider
        expiry_str = self.cp.expiry_combo.currentText()
        actual_dte = self._compute_dte(expiry_str)
        if actual_dte is not None:
            self.cp.set_dte_from_expiry(actual_dte)

        # Update the option chain widget
        dte = self.cp.dte_override
        self.chain_widget.set_market_params(self._spot, self._rate, self._div_yield, dte)
        self.chain_widget.load_chain(chain)

        # If a strategy template is selected, auto-build it
        from ui.control_panel import _STRATEGY_TEMPLATES
        current_template = self.cp.strategy_combo.currentText()
        key = _STRATEGY_TEMPLATES.get(current_template)
        if key:
            self._build_strategy_from_template(key)
        else:
            self._on_reset_view()

    def _on_chain_error(self, msg: str):
        self.cp.set_status(f"\u2717 Chain: {msg}", Colors.ACCENT_RED)

    @staticmethod
    def _compute_dte(expiry_str: str) -> float | None:
        """Parse an expiry string and return DTE from today."""
        try:
            exp = datetime.strptime(expiry_str, "%Y-%m-%d").date()
            return max(float((exp - date.today()).days), 0.0)
        except (ValueError, TypeError):
            return None

    # ----- Expiry changed --------------------------------------------------

    def _on_expiry_changed(self, expiry: str):
        if self._ticker and expiry:
            self._fetch_chain(self._ticker, expiry)

    # ----- Option chain click → add leg ------------------------------------

    def _on_chain_leg_added(self, leg_data: dict):
        """A bid/ask cell was clicked in the option chain."""
        expiry = self.cp.expiry_combo.currentText() or "2026-12-31"
        leg = OptionLeg(
            option_type=leg_data["type"],
            direction=leg_data["dir"],
            strike=leg_data["strike"],
            expiry_date=expiry,
            quantity=1,
            iv=leg_data["iv"],
            premium=leg_data.get("premium", 0.0),
        )
        self._strategy.add_leg(leg)
        self._sync_legs_to_ui()
        self._refresh_all_charts()

    # ----- Remove / clear legs ---------------------------------------------

    def _on_remove_leg(self, index: int):
        self._strategy.remove_leg(index)
        self._sync_legs_to_ui()
        self._refresh_all_charts()

    def _on_clear_legs(self):
        self._strategy.clear_legs()
        self._sync_legs_to_ui()
        self._refresh_all_charts()
        self._on_reset_view()

    # ----- Strategy template -----------------------------------------------

    def _on_strategy_template(self, key):
        if key is None:
            return
        if self._chain is not None and not self._chain.empty:
            self._build_strategy_from_template(key)

    def _build_strategy_from_template(self, template_key: str):
        """Auto-populate legs from chain data based on ATM strikes."""
        if self._chain is None:
            return

        calls = self._chain[self._chain["option_type"] == "call"].sort_values("strike")
        puts = self._chain[self._chain["option_type"] == "put"].sort_values("strike")

        if calls.empty:
            return

        atm_idx = (calls["strike"] - self._spot).abs().idxmin()
        atm_strike = calls.loc[atm_idx, "strike"]
        atm_iv = calls.loc[atm_idx, "iv"]

        all_strikes = sorted(calls["strike"].unique())
        atm_pos = min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i] - atm_strike))

        def _strike_at(offset):
            idx = max(0, min(len(all_strikes) - 1, atm_pos + offset))
            return all_strikes[idx]

        def _iv_for(strike, opt_type="call"):
            df = calls if opt_type == "call" else puts
            row = df[df["strike"] == strike]
            return float(row["iv"].iloc[0]) if not row.empty else atm_iv

        def _mid_for(strike, opt_type="call"):
            df = calls if opt_type == "call" else puts
            row = df[df["strike"] == strike]
            return float(row["mid"].iloc[0]) if not row.empty and "mid" in row.columns else 0.0

        if template_key in ("long_call", "long_put"):
            strikes = [atm_strike]
            ot = "call" if "call" in template_key else "put"
            ivs = [_iv_for(atm_strike, ot)]
            premiums = [_mid_for(atm_strike, ot)]
        elif template_key == "bull_call_spread":
            s_low, s_high = _strike_at(-2), _strike_at(2)
            strikes = [s_low, s_high]
            ivs = [_iv_for(s_low), _iv_for(s_high)]
            premiums = [_mid_for(s_low), _mid_for(s_high)]
        elif template_key == "bear_put_spread":
            s_high, s_low = _strike_at(2), _strike_at(-2)
            strikes = [s_high, s_low]
            ivs = [_iv_for(s_high, "put"), _iv_for(s_low, "put")]
            premiums = [_mid_for(s_high, "put"), _mid_for(s_low, "put")]
        elif template_key == "long_straddle":
            strikes = [atm_strike, atm_strike]
            ivs = [_iv_for(atm_strike, "call"), _iv_for(atm_strike, "put")]
            premiums = [_mid_for(atm_strike, "call"), _mid_for(atm_strike, "put")]
        elif template_key == "long_strangle":
            s_low, s_high = _strike_at(-3), _strike_at(3)
            strikes = [s_low, s_high]
            ivs = [_iv_for(s_low, "put"), _iv_for(s_high, "call")]
            premiums = [_mid_for(s_low, "put"), _mid_for(s_high, "call")]
        elif template_key == "iron_condor":
            strikes = [_strike_at(-5), _strike_at(-3), _strike_at(3), _strike_at(5)]
            ivs = [
                _iv_for(strikes[0], "put"), _iv_for(strikes[1], "put"),
                _iv_for(strikes[2], "call"), _iv_for(strikes[3], "call"),
            ]
            premiums = [
                _mid_for(strikes[0], "put"), _mid_for(strikes[1], "put"),
                _mid_for(strikes[2], "call"), _mid_for(strikes[3], "call"),
            ]
        elif template_key == "butterfly":
            strikes = [_strike_at(-3), atm_strike, _strike_at(3)]
            ivs = [_iv_for(s) for s in strikes]
            premiums = [_mid_for(s) for s in strikes]
        else:
            return

        expiry = self.cp.expiry_combo.currentText() or "2026-12-31"
        self._strategy = Strategy.from_template(
            template_key, self._spot, strikes, expiry, ivs, premiums,
            r=self._rate, q=self._div_yield,
        )

        self._sync_legs_to_ui()
        self._refresh_all_charts()
        self._on_reset_view()

    # ----- Sync strategy legs → legs table UI ------------------------------

    def _sync_legs_to_ui(self):
        """Push current strategy legs into the sidebar legs table."""
        legs_data = []
        for leg in self._strategy.legs:
            legs_data.append({
                "type": leg.option_type,
                "dir": leg.direction,
                "strike": leg.strike,
                "qty": leg.quantity,
                "iv": leg.iv,
                "expiry": leg.expiry_date,
            })
        self.cp.populate_legs_table(legs_data)

    def _on_legs_edited(self):
        """User edited a cell widget in the legs table — read back & update.

        We read the editable table widgets and mutate the strategy's legs
        in-place, then refresh all charts.
        """
        for row in range(self.legs_table_row_count()):
            if row >= len(self._strategy.legs):
                break
            leg = self._strategy.legs[row]

            type_combo = self.cp.legs_table.cellWidget(row, 0)
            dir_combo = self.cp.legs_table.cellWidget(row, 1)
            strike_spin = self.cp.legs_table.cellWidget(row, 2)
            qty_spin = self.cp.legs_table.cellWidget(row, 3)
            iv_spin = self.cp.legs_table.cellWidget(row, 4)

            if not all([type_combo, dir_combo, strike_spin, qty_spin, iv_spin]):
                continue

            leg.option_type = type_combo.currentText()
            leg.direction = dir_combo.currentText()
            leg.strike = strike_spin.value()
            leg.quantity = qty_spin.value()
            leg.iv = iv_spin.value() / 100.0

        self._refresh_all_charts()

    def legs_table_row_count(self) -> int:
        return self.cp.legs_table.rowCount()

    # ----- Parameter changes (sliders) -------------------------------------

    def _on_params_changed(self):
        """Slider moved — update pyqtgraph immediately, debounce Plotly."""
        self._strategy.risk_free_rate = self.cp.rate_override
        self._update_pnl()
        self._update_greeks()
        self._update_sensitivity()
        self._surface_timer.start()  # debounce Plotly
        self._update_summary()

    # ----- DTE offset computation ------------------------------------------

    def _dte_offset(self) -> float:
        """How many days the DTE slider has been *reduced* from the nearest
        expiry's actual DTE.  Used to shift each leg's individual T.

        Example: if the nearest expiry is 45 days out and the slider reads
        30 days, the offset is 15 — meaning "show me the strategy as if
        15 fewer calendar days remain for every leg."
        """
        expiry_str = self.cp.expiry_combo.currentText()
        actual = self._compute_dte(expiry_str)
        slider_dte = self.cp.dte_override
        if actual is not None:
            return max(actual - slider_dte, 0.0)
        return 0.0

    # ----- Chart updates & panning -----------------------------------------

    def _on_reset_view(self):
        spot = self._spot
        if spot <= 0:
            return
            
        # PnL Chart
        vb_pnl = self.pnl_chart.getPlotItem().vb
        vb_pnl.setXRange(spot * 0.85, spot * 1.15, padding=0)
        vb_pnl.enableAutoRange(axis=pg.ViewBox.YAxis)
        vb_pnl.setAutoVisible(x=False, y=True)
        
        # Greek Chart
        for p in self.greek_chart._plots:
            p.vb.setXRange(spot * 0.85, spot * 1.15, padding=0)
            p.vb.enableAutoRange(axis=pg.ViewBox.YAxis)
            p.vb.setAutoVisible(x=False, y=True)
            
        if hasattr(self.greek_chart, "vb_gamma"):
            self.greek_chart.vb_gamma.enableAutoRange(axis=pg.ViewBox.YAxis)
            self.greek_chart.vb_gamma.setAutoVisible(x=False, y=True)

    def _refresh_all_charts(self):
        self._update_pnl()
        self._update_greeks()
        self._update_surface()
        self._update_sensitivity()
        self._update_summary()

    def _spot_range(self):
        """Generate a fixed ±50% range around spot with 500 points."""
        if self._spot <= 0:
            return np.linspace(50, 150, 500)
        low = self._spot * 0.50
        high = self._spot * 1.50
        return np.linspace(low, high, 500)

    def _update_pnl(self):
        if not self._strategy.legs:
            return
        spot_range = self._spot_range()
        iv_shift = self.cp.iv_shift
        offset = self._dte_offset()

        # T+0 (today — per-leg T from expiry_date, offset=0)
        pnl_t0 = self._strategy.total_pnl(
            spot_range, iv_shift=iv_shift, dte_offset_days=offset,
        )
        # T+N (Target Days Passed)
        target_days = self.cp.target_dte_override
        pnl_mid = self._strategy.total_pnl(
            spot_range, iv_shift=iv_shift,
            dte_offset_days=offset + target_days,
        )
        # At expiry (evaluated exactly when the nearest-term leg expires)
        nearest_dte = self._strategy.nearest_dte()
        pnl_exp = self._strategy.total_pnl(
            spot_range, iv_shift=iv_shift, dte_offset_days=nearest_dte,
        )

        self.pnl_chart.update_plot(
            spot_range, pnl_exp, pnl_t0, pnl_mid,
            spot_price=self._spot if self._spot > 0 else None,
            target_label=f"T+{int(target_days)}",
        )

    def _update_greeks(self):
        if not self._strategy.legs:
            return
        spot_range = self._spot_range()
        iv_shift = self.cp.iv_shift
        offset = self._dte_offset()

        greeks = {}
        for name in GREEK_NAMES:
            greeks[name] = self._strategy.total_greek(
                name, spot_range, iv_shift=iv_shift, dte_offset_days=offset,
            )

        self.greek_chart.update_plot(
            spot_range, greeks,
            spot_price=self._spot if self._spot > 0 else None,
        )

    def _update_surface(self):
        """Render the 3D Plotly surface (debounced)."""
        if not self._strategy.legs:
            return
        greek_name = self.cp.surface_greek or "delta"
        iv_shift = self.cp.iv_shift
        offset = self._dte_offset()

        spot_range = self._spot_range()
        max_dte = max(self.cp.dte_override, 5)
        dte_range = np.linspace(1, max_dte, 40)

        z_rows = []
        for d in dte_range:
            extra_offset = max_dte - d  # offset increases as DTE decreases
            row = self._strategy.total_greek(
                greek_name, spot_range, iv_shift=iv_shift,
                dte_offset_days=offset + extra_offset,
            )
            z_rows.append(row)
        Z = np.array(z_rows)

        self.surface_chart.update_surface(
            x=spot_range, y=dte_range, z=Z,
            x_label="Underlying Price",
            y_label="DTE (days)",
            z_label=greek_name.capitalize(),
        )

    def _update_sensitivity(self):
        """Update the sensitivity chart based on selected mode."""
        if not self._strategy.legs:
            return
        mode = self.cp.sensitivity_mode or "delta_iv"
        offset = self._dte_offset()

        if mode.endswith("_iv"):
            greek_name = mode.split("_")[0]
            iv_shifts = np.linspace(-0.20, 0.20, 80)
            curves = {}
            for pct in [-10, -5, 0, 5, 10]:
                s = self._spot * (1 + pct / 100.0) if self._spot > 0 else 100 + pct
                values = []
                for shift in iv_shifts:
                    g = self._strategy.total_greek(
                        greek_name, np.array([s]), iv_shift=shift,
                        dte_offset_days=offset,
                    )
                    values.append(float(g[0]))
                label = f"S={s:.0f}" if pct != 0 else f"S={s:.0f} (ATM)"
                curves[label] = np.array(values)
            self.sensitivity_chart.update_plot(
                iv_shifts * 100, curves,
                x_label="IV Shift (%)",
                y_label=f"{greek_name.capitalize()} Value",
            )
        elif mode.endswith("_dte"):
            greek_name = mode.split("_")[0]
            max_dte = max(self.cp.dte_override, 30)
            dte_range = np.linspace(1, max_dte, 80)
            curves = {}
            for pct in [-10, -5, 0, 5, 10]:
                s = self._spot * (1 + pct / 100.0) if self._spot > 0 else 100 + pct
                values = []
                for d in dte_range:
                    extra = max_dte - d
                    g = self._strategy.total_greek(
                        greek_name, np.array([s]), iv_shift=self.cp.iv_shift,
                        dte_offset_days=offset + extra,
                    )
                    values.append(float(g[0]))
                label = f"S={s:.0f}" if pct != 0 else f"S={s:.0f} (ATM)"
                curves[label] = np.array(values)
            self.sensitivity_chart.update_plot(
                dte_range, curves,
                x_label="DTE (days)",
                y_label=f"{greek_name.capitalize()} Value",
            )

    def _update_summary(self):
        if not self._strategy.legs:
            self.summary_panel.clear_summary()
            return
        
        iv_shift = self.cp.iv_shift
        offset = self._dte_offset()
        S = np.array([self._spot])
        
        greeks = {}
        for name in ["delta", "gamma", "theta", "vega", "rho"]:
            val = self._strategy.total_greek(
                name, S, iv_shift=iv_shift, dte_offset_days=offset,
            )
            v = float(val[0])
            
            # Apply standard display scaling
            if name == "theta":
                v /= 365.0
            elif name in ("vega", "rho"):
                v /= 100.0
                
            greeks[name] = v

        nearest_dte = self._strategy.nearest_dte()
        
        # Evaluate at strikes + extreme bounds (0 and 1e6) to find true max profit/loss
        strikes = [leg.strike for leg in self._strategy.legs]
        test_spots = np.array([0.0, 1e6] + strikes)
        pnl_test = self._strategy.total_pnl(test_spots, iv_shift=iv_shift, dte_offset_days=nearest_dte)
        
        pnl_zero = float(pnl_test[0])
        pnl_inf = float(pnl_test[1])
        local_pnl = np.concatenate(([pnl_zero], pnl_test[2:]))
        
        max_profit_val = float(np.max(local_pnl))
        max_loss_val = float(np.min(local_pnl))
        
        if pnl_inf > 1e5:
            max_profit_str = "Infinite"
        else:
            max_profit_val = max(max_profit_val, pnl_inf)
            max_profit_str = f"${max_profit_val:+,.0f}"
            
        if pnl_inf < -1e5:
            max_loss_str = "Infinite"
        else:
            max_loss_val = min(max_loss_val, pnl_inf)
            max_loss_str = f"${max_loss_val:+,.0f}"

        # Original spot range for breakevens
        spot_range = self._spot_range()
        pnl_exp = self._strategy.total_pnl(spot_range, iv_shift=iv_shift, dte_offset_days=nearest_dte)

        breakevens = []
        for i in range(len(pnl_exp) - 1):
            if pnl_exp[i] * pnl_exp[i + 1] < 0:
                x0, x1 = spot_range[i], spot_range[i + 1]
                y0, y1 = pnl_exp[i], pnl_exp[i + 1]
                be = x0 - y0 * (x1 - x0) / (y1 - y0)
                breakevens.append(be)

        self.summary_panel.update_summary(greeks, max_profit_str, max_loss_str, breakevens)
