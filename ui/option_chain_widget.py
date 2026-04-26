"""
ui/option_chain_widget.py

Interactive Brokers / TradingView–style option chain table.
Layout: Calls (Bid, Ask, IV%, Δ) | Strike | Puts (Bid, Ask, IV%, Δ)

Clicking a Bid cell  → add SHORT leg for that option type/strike.
Clicking an Ask cell → add LONG  leg for that option type/strike.

Features:
- Symmetrical column order (Bid always before Ask on both sides).
- OTM-blended IV: uses the OTM option's IV as the unified IV per strike,
  enforcing visual put-call parity across the chain.
- Auto-scroll to ATM on every load.
- Compact ResizeToContents on data columns, Stretch on Strike.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor, QBrush

import pandas as pd
import numpy as np

from engine import black_scholes as bsm
from ui.theme import Colors, Fonts

# Column layout — symmetrical: Calls then Puts, Bid always before Ask.
# Calls: Bid | Ask | IV% | Δ  |  Strike  |  Puts: Bid | Ask | IV% | Δ
_HEADERS = ["C Bid", "C Ask", "C IV%", "C Δ", "Strike", "P Bid", "P Ask", "P IV%", "P Δ"]
_COL_C_BID   = 0
_COL_C_ASK   = 1
_COL_C_IV    = 2
_COL_C_DELTA = 3
_COL_STRIKE  = 4
_COL_P_BID   = 5
_COL_P_ASK   = 6
_COL_P_IV    = 7
_COL_P_DELTA = 8


class OptionChainWidget(QWidget):
    """IB-style option chain with click-to-add-leg interaction.

    Signals
    -------
    leg_added(dict)
        Emitted when the user clicks a bid/ask cell.
        Payload: {'type': 'call'|'put', 'dir': 'long'|'short',
                  'strike': float, 'iv': float, 'premium': float}
    """

    leg_added = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._chain_df: pd.DataFrame | None = None
        self._spot = 0.0
        self._rate = 0.05
        self._div_yield = 0.0
        self._T = 30 / 365.0
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Header labels
        header_row = QHBoxLayout()
        calls_label = QLabel("CALLS")
        calls_label.setStyleSheet(
            f"color: {Colors.ACCENT_GREEN}; font-weight: 600; font-size: {Fonts.SIZE_XS}px;"
        )
        calls_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        strike_label = QLabel("STRIKE")
        strike_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-weight: 600; font-size: {Fonts.SIZE_XS}px;"
        )
        strike_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        puts_label = QLabel("PUTS")
        puts_label.setStyleSheet(
            f"color: {Colors.ACCENT_RED}; font-weight: 600; font-size: {Fonts.SIZE_XS}px;"
        )
        puts_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        header_row.addWidget(calls_label)
        header_row.addWidget(strike_label)
        header_row.addWidget(puts_label)
        layout.addLayout(header_row)

        # Table
        self.table = QTableWidget(0, len(_HEADERS))
        self.table.setHorizontalHeaderLabels(_HEADERS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setAlternatingRowColors(False)
        self.table.setShowGrid(False)

        # Font
        mono = QFont(Fonts.MONO.split(",")[0].strip(), Fonts.SIZE_XS)
        self.table.setFont(mono)

        # Column sizing — compact data cols, Stretch for Strike
        header = self.table.horizontalHeader()
        for col in range(len(_HEADERS)):
            if col == _COL_STRIKE:
                header.setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)
            else:
                header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)

        # Minimal row height
        self.table.verticalHeader().setDefaultSectionSize(22)

        # Style
        self.table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {Colors.BG_PRIMARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                gridline-color: transparent;
                font-family: {Fonts.MONO};
                font-size: {Fonts.SIZE_XS}px;
            }}
            QTableWidget::item {{
                padding: 1px 4px;
                border-bottom: 1px solid {Colors.BORDER};
            }}
            QHeaderView::section {{
                background-color: {Colors.BG_ELEVATED};
                color: {Colors.TEXT_SECONDARY};
                border: none;
                border-bottom: 1px solid {Colors.BORDER};
                padding: 3px 4px;
                font-weight: 600;
                font-size: 9px;
            }}
        """)

        # Click handler
        self.table.cellClicked.connect(self._on_cell_clicked)

        layout.addWidget(self.table)

    def set_market_params(self, spot: float, rate: float, div_yield: float, dte_days: float):
        """Update market parameters used for Delta computation."""
        self._spot = spot
        self._rate = rate
        self._div_yield = div_yield
        self._T = max(dte_days / 365.0, 1e-7)

    def load_chain(self, chain_df: pd.DataFrame):
        """Populate the table from a cleaned chain DataFrame."""
        self._chain_df = chain_df
        self._render()

    @staticmethod
    def _otm_blended_iv(
        strike: float, spot: float,
        c_iv: float, p_iv: float,
    ) -> float:
        """Synthesize a smoothed IV using the OTM option's IV.

        - strike < spot  → OTM put  → use put IV
        - strike > spot  → OTM call → use call IV
        - strike ≈ spot  → average of both
        - If one side is missing (0), fall back to the other.
        """
        if c_iv <= 0 and p_iv <= 0:
            return 0.0
        if c_iv <= 0:
            return p_iv
        if p_iv <= 0:
            return c_iv

        if spot <= 0:
            return (c_iv + p_iv) / 2.0

        # OTM selection
        pct_diff = (strike - spot) / spot
        if pct_diff < -0.001:
            # Below spot → OTM put
            return p_iv
        elif pct_diff > 0.001:
            # Above spot → OTM call
            return c_iv
        else:
            # Near ATM → average
            return (c_iv + p_iv) / 2.0

    def _render(self):
        if self._chain_df is None or self._chain_df.empty:
            self.table.setRowCount(0)
            return

        df = self._chain_df
        calls = df[df["option_type"] == "call"].set_index("strike")
        puts = df[df["option_type"] == "put"].set_index("strike")
        all_strikes = sorted(set(calls.index) | set(puts.index))

        self.table.setRowCount(len(all_strikes))

        # Brushes
        bid_brush = QBrush(QColor(Colors.ACCENT_RED + "25"))
        ask_brush = QBrush(QColor(Colors.ACCENT_GREEN + "25"))
        strike_brush = QBrush(QColor(Colors.BG_ELEVATED))
        atm_brush = QBrush(QColor(Colors.ACCENT_BLUE + "25"))

        # Find ATM
        if self._spot > 0:
            atm_strike = min(all_strikes, key=lambda s: abs(s - self._spot))
        else:
            atm_strike = None

        for row_idx, strike in enumerate(all_strikes):
            is_atm = (strike == atm_strike)

            # ---- Extract raw data ----
            if strike in calls.index:
                c = calls.loc[strike]
                if isinstance(c, pd.DataFrame):
                    c = c.iloc[0]
                c_bid = c.get("bid", 0.0)
                c_ask = c.get("ask", 0.0)
                c_iv_raw = c.get("iv", 0.0)
            else:
                c_bid = c_ask = c_iv_raw = 0.0

            if strike in puts.index:
                p = puts.loc[strike]
                if isinstance(p, pd.DataFrame):
                    p = p.iloc[0]
                p_bid = p.get("bid", 0.0)
                p_ask = p.get("ask", 0.0)
                p_iv_raw = p.get("iv", 0.0)
            else:
                p_bid = p_ask = p_iv_raw = 0.0

            # ---- Blended OTM IV ----
            blended_iv = self._otm_blended_iv(strike, self._spot, c_iv_raw, p_iv_raw)

            # ---- Compute deltas using blended IV ----
            iv_for_calc = max(blended_iv, 0.001)
            S = self._spot if self._spot > 0 else strike
            c_delta = float(bsm.delta(S, strike, self._T, self._rate, iv_for_calc,
                                       self._div_yield, "call"))
            p_delta = float(bsm.delta(S, strike, self._T, self._rate, iv_for_calc,
                                       self._div_yield, "put"))

            # ---- Build row: symmetrical order ----
            # Cols: C Bid | C Ask | C IV% | C Δ | Strike | P Bid | P Ask | P IV% | P Δ
            items = [
                (f"{c_bid:.2f}",           bid_brush),                               # 0: C Bid
                (f"{c_ask:.2f}",           ask_brush),                               # 1: C Ask
                (f"{blended_iv*100:.1f}%", None),                                    # 2: C IV%
                (f"{c_delta:.3f}",         None),                                    # 3: C Δ
                (f"{strike:.1f}",          atm_brush if is_atm else strike_brush),   # 4: Strike
                (f"{p_bid:.2f}",           bid_brush),                               # 5: P Bid
                (f"{p_ask:.2f}",           ask_brush),                               # 6: P Ask
                (f"{blended_iv*100:.1f}%", None),                                    # 7: P IV%
                (f"{p_delta:.3f}",         None),                                    # 8: P Δ
            ]

            for col_idx, (text, bg) in enumerate(items):
                item = QTableWidgetItem(text)
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )

                # Color coding
                if col_idx == _COL_STRIKE:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    if is_atm:
                        item.setForeground(QBrush(QColor(Colors.ACCENT_BLUE)))
                    else:
                        item.setForeground(QBrush(QColor(Colors.TEXT_PRIMARY)))
                elif col_idx in (_COL_C_BID, _COL_P_BID):
                    item.setForeground(QBrush(QColor(Colors.ACCENT_RED)))
                elif col_idx in (_COL_C_ASK, _COL_P_ASK):
                    item.setForeground(QBrush(QColor(Colors.ACCENT_GREEN)))
                elif col_idx in (_COL_C_IV, _COL_P_IV):
                    item.setForeground(QBrush(QColor(Colors.TEXT_SECONDARY)))
                elif col_idx == _COL_C_DELTA:
                    item.setForeground(QBrush(QColor(Colors.ACCENT_BLUE)))
                elif col_idx == _COL_P_DELTA:
                    item.setForeground(QBrush(QColor(Colors.ACCENT_RED)))

                if bg is not None:
                    item.setBackground(bg)

                # Store data for click handler
                item.setData(Qt.ItemDataRole.UserRole, {
                    "strike": strike,
                    "blended_iv": blended_iv,
                    "c_iv": c_iv_raw, "p_iv": p_iv_raw,
                    "c_bid": c_bid, "c_ask": c_ask,
                    "p_bid": p_bid, "p_ask": p_ask,
                })

                self.table.setItem(row_idx, col_idx, item)

        # ---- Auto-scroll to ATM (delayed to allow table layout) ----
        if atm_strike is not None:
            atm_row = all_strikes.index(atm_strike)
            QTimer.singleShot(50, lambda r=atm_row: self.table.scrollTo(
                self.table.model().index(r, _COL_STRIKE),
                QAbstractItemView.ScrollHint.PositionAtCenter,
            ))

    def _on_cell_clicked(self, row: int, col: int):
        """Handle click → add leg.  Uses blended IV for the leg."""
        item = self.table.item(row, col)
        if item is None:
            return

        data = item.data(Qt.ItemDataRole.UserRole)
        if data is None:
            return

        strike = data["strike"]
        blended_iv = data["blended_iv"]

        # Determine option type and direction from column
        if col == _COL_C_BID:
            # Selling a call → short
            self.leg_added.emit({
                "type": "call", "dir": "short", "strike": strike,
                "iv": blended_iv, "premium": data["c_bid"],
            })
        elif col == _COL_C_ASK:
            # Buying a call → long
            self.leg_added.emit({
                "type": "call", "dir": "long", "strike": strike,
                "iv": blended_iv, "premium": data["c_ask"],
            })
        elif col == _COL_P_BID:
            # Selling a put → short
            self.leg_added.emit({
                "type": "put", "dir": "short", "strike": strike,
                "iv": blended_iv, "premium": data["p_bid"],
            })
        elif col == _COL_P_ASK:
            # Buying a put → long
            self.leg_added.emit({
                "type": "put", "dir": "long", "strike": strike,
                "iv": blended_iv, "premium": data["p_ask"],
            })
        # Clicking strike/IV/delta columns does nothing
