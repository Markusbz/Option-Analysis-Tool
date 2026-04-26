"""
ui/control_panel.py

Sidebar control panel: ticker input, strategy picker, expiry selector,
EDITABLE active legs table, and "What-If" parameter sliders.
The Option Chain Widget lives in the main window (center area);
this panel shows the active strategy legs and configuration controls.
"""

from __future__ import annotations

from datetime import datetime

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QSlider, QTableWidget,
    QTableWidgetItem, QHeaderView, QFrame, QAbstractItemView,
    QScrollArea, QSpinBox, QDoubleSpinBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QBrush

from ui.theme import Colors, Fonts


# Available strategy templates (display name → engine key)
_STRATEGY_TEMPLATES = {
    "Custom (click chain)": None,
    "Long Call": "long_call",
    "Long Put": "long_put",
    "Bull Call Spread": "bull_call_spread",
    "Bear Put Spread": "bear_put_spread",
    "Long Straddle": "long_straddle",
    "Long Strangle": "long_strangle",
    "Iron Condor": "iron_condor",
    "Butterfly": "butterfly",
}


class _SectionLabel(QLabel):
    """Styled uppercase section header."""
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setProperty("class", "section-title")
        self.setFont(QFont(Fonts.UI.split(",")[0].strip(), Fonts.SIZE_XS))


class _SliderSpinBox(QWidget):
    """Composite slider + spinbox with two-way synchronization."""

    valueChanged = pyqtSignal(float)

    def __init__(self, label: str, min_val: float, max_val: float,
                 default: float, step: float = 0.01, decimals: int = 2,
                 suffix: str = "", parent=None):
        super().__init__(parent)
        self._min = min_val
        self._max = max_val
        self._step = step

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        layout.setSpacing(6)

        # Label
        self._label = QLabel(label)
        self._label.setFixedWidth(52)
        self._label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_XS}px;")

        # Slider (integer ticks)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(int((max_val - min_val) / step))
        self._slider.setValue(int((default - min_val) / step))

        # SpinBox
        self._spinbox = QDoubleSpinBox()
        self._spinbox.setRange(min_val, max_val)
        self._spinbox.setDecimals(decimals)
        self._spinbox.setSingleStep(step)
        self._spinbox.setValue(default)
        self._spinbox.setSuffix(suffix)
        self._spinbox.setFixedWidth(65)
        self._spinbox.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self._spinbox.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        # Two-way sync
        self._internal_update = False

        self._slider.valueChanged.connect(self._on_slider_changed)
        self._spinbox.valueChanged.connect(self._on_spinbox_changed)

        layout.addWidget(self._label)
        layout.addWidget(self._slider, stretch=1)
        layout.addWidget(self._spinbox)

    def _on_slider_changed(self, tick: int):
        if self._internal_update:
            return
        val = self._min + tick * self._step
        self._internal_update = True
        self._spinbox.setValue(val)
        self._internal_update = False
        self.valueChanged.emit(val)

    def _on_spinbox_changed(self, val: float):
        if self._internal_update:
            return
        tick = int(round((val - self._min) / self._step))
        self._internal_update = True
        self._slider.setValue(tick)
        self._internal_update = False
        self.valueChanged.emit(val)

    def value(self) -> float:
        return self._spinbox.value()

    def reset(self, val: float):
        """Reset slider and spinbox to val, clamping to [min, max]."""
        val = max(self._min, min(val, self._max))
        self._internal_update = True
        self._spinbox.setValue(val)
        tick = int(round((val - self._min) / self._step))
        self._slider.setValue(tick)
        self._internal_update = False

    def set_maximum(self, max_val: float):
        """Dynamically update the maximum for both widgets."""
        self._max = max_val
        self._internal_update = True
        self._spinbox.setMaximum(max_val)
        self._slider.setMaximum(int((max_val - self._min) / self._step))
        self._internal_update = False


class ControlPanel(QWidget):
    """Sidebar control panel with editable legs table.

    Signals
    -------
    fetch_requested(str)
        Emitted when the user clicks Fetch.
    strategy_changed(str | None)
        Emitted when the strategy dropdown changes.
    expiry_changed(str)
        Emitted when the expiry dropdown changes.
    legs_changed()
        Emitted when a leg is edited in the table.
    params_changed()
        Emitted when any What-If slider moves.
    clear_legs_requested()
        Emitted when user clicks Clear All.
    remove_leg_requested(int)
        Emitted when user removes a specific leg by index.
    """

    fetch_requested = pyqtSignal(str)
    strategy_changed = pyqtSignal(object)
    expiry_changed = pyqtSignal(str)
    legs_changed = pyqtSignal()
    params_changed = pyqtSignal()
    clear_legs_requested = pyqtSignal()
    remove_leg_requested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(300)
        self.setStyleSheet(f"background-color: {Colors.BG_SURFACE};")
        self._populating = False  # guard to suppress signals during populate
        self._build_ui()

    def _build_ui(self):
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QWidget()
        self._layout = QVBoxLayout(inner)
        self._layout.setContentsMargins(12, 10, 12, 10)
        self._layout.setSpacing(4)

        # --- App title ---
        title = QLabel("Options Analysis")
        title.setFont(QFont(Fonts.UI.split(",")[0].strip(), Fonts.SIZE_TITLE, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY};")
        self._layout.addWidget(title)

        subtitle = QLabel("Strategy Builder & Greek Visualizer")
        subtitle.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_XS}px;")
        self._layout.addWidget(subtitle)
        self._layout.addSpacing(6)

        # --- Ticker ---
        self._layout.addWidget(_SectionLabel("TICKER"))
        ticker_row = QHBoxLayout()
        ticker_row.setSpacing(6)
        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("e.g. SPY, AAPL, QQQ")
        self.ticker_input.setText("SPY")
        self.ticker_input.returnPressed.connect(self._on_fetch)
        ticker_row.addWidget(self.ticker_input, stretch=1)

        self.fetch_btn = QPushButton("Fetch")
        self.fetch_btn.setFixedWidth(60)
        self.fetch_btn.clicked.connect(self._on_fetch)
        ticker_row.addWidget(self.fetch_btn)
        self._layout.addLayout(ticker_row)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_XS}px;")
        self.status_label.setWordWrap(True)
        self._layout.addWidget(self.status_label)
        self._layout.addSpacing(4)

        # --- Spot + Rate display ---
        info_row = QHBoxLayout()
        info_row.setSpacing(8)
        self.spot_label = QLabel("Spot: \u2014")
        self.spot_label.setFont(QFont(Fonts.MONO.split(",")[0].strip(), Fonts.SIZE_XS))
        self.spot_label.setStyleSheet(f"color: {Colors.ACCENT_GREEN};")
        info_row.addWidget(self.spot_label)

        self.rate_label = QLabel("r: \u2014")
        self.rate_label.setFont(QFont(Fonts.MONO.split(",")[0].strip(), Fonts.SIZE_XS))
        self.rate_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        info_row.addWidget(self.rate_label)

        self.div_label = QLabel("q: \u2014")
        self.div_label.setFont(QFont(Fonts.MONO.split(",")[0].strip(), Fonts.SIZE_XS))
        self.div_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        info_row.addWidget(self.div_label)
        self._layout.addLayout(info_row)
        self._layout.addSpacing(4)

        # --- Expiration ---
        self._layout.addWidget(_SectionLabel("EXPIRATION"))
        self.expiry_combo = QComboBox()
        self.expiry_combo.currentTextChanged.connect(
            lambda t: self.expiry_changed.emit(t) if t else None
        )
        self._layout.addWidget(self.expiry_combo)
        self._layout.addSpacing(4)

        # --- Strategy template ---
        self._layout.addWidget(_SectionLabel("STRATEGY"))
        self.strategy_combo = QComboBox()
        for display_name in _STRATEGY_TEMPLATES:
            self.strategy_combo.addItem(display_name)
        self.strategy_combo.currentTextChanged.connect(self._on_strategy_changed)
        self._layout.addWidget(self.strategy_combo)
        self._layout.addSpacing(4)

        # --- Active Legs (EDITABLE) ---
        self._layout.addWidget(_SectionLabel("ACTIVE LEGS"))

        self.legs_table = QTableWidget(0, 6)
        self.legs_table.setHorizontalHeaderLabels(["Type", "Dir", "K", "Qty", "IV%", "Exp"])
        hdr = self.legs_table.horizontalHeader()
        # Resize first 5 columns to contents, stretch the last column (Exp)
        for i in range(5):
            hdr.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.legs_table.verticalHeader().setVisible(False)
        self.legs_table.verticalHeader().setDefaultSectionSize(32)
        self.legs_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.legs_table.setMaximumHeight(180)
        self.legs_table.setFont(QFont(Fonts.MONO.split(",")[0].strip(), Fonts.SIZE_XS))
        self.legs_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {Colors.BG_PRIMARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                gridline-color: {Colors.BORDER};
            }}
            QTableWidget::item {{
                padding: 1px 3px;
            }}
            QHeaderView::section {{
                background-color: {Colors.BG_ELEVATED};
                color: {Colors.TEXT_SECONDARY};
                border: none;
                border-bottom: 1px solid {Colors.BORDER};
                padding: 2px 4px;
                font-weight: 600;
                font-size: 9px;
            }}
        """)
        self._layout.addWidget(self.legs_table)

        leg_btn_row = QHBoxLayout()
        leg_btn_row.setSpacing(6)

        self.remove_leg_btn = QPushButton("\u2212 Remove")
        self.remove_leg_btn.setProperty("class", "danger")
        self.remove_leg_btn.clicked.connect(self._on_remove_leg)
        leg_btn_row.addWidget(self.remove_leg_btn)

        self.clear_legs_btn = QPushButton("Clear All")
        self.clear_legs_btn.setProperty("class", "secondary")
        self.clear_legs_btn.clicked.connect(lambda: self.clear_legs_requested.emit())
        leg_btn_row.addWidget(self.clear_legs_btn)
        self._layout.addLayout(leg_btn_row)

        hint = QLabel("Click Bid/Ask in chain, or edit cells directly")
        hint.setStyleSheet(f"color: {Colors.TEXT_DISABLED}; font-size: 9px; font-style: italic;")
        self._layout.addWidget(hint)
        self._layout.addSpacing(6)

        # --- What-If sliders ---
        self._layout.addWidget(_SectionLabel("WHAT-IF SCENARIO"))

        self.iv_slider = _SliderSpinBox("IV Shift", -50.0, 50.0, 0.0, step=0.5, decimals=1, suffix="%")
        self.iv_slider.valueChanged.connect(lambda _: self.params_changed.emit())
        self._layout.addWidget(self.iv_slider)

        self.dte_slider = _SliderSpinBox("DTE", 0.0, 365.0, 30.0, step=1.0, decimals=0, suffix=" d")
        self.dte_slider.valueChanged.connect(lambda _: self.params_changed.emit())
        self._layout.addWidget(self.dte_slider)

        self.target_dte_slider = _SliderSpinBox("Target", 0.0, 365.0, 15.0, step=1.0, decimals=0, suffix=" d")
        self.target_dte_slider.valueChanged.connect(lambda _: self.params_changed.emit())
        self._layout.addWidget(self.target_dte_slider)

        self.rate_slider = _SliderSpinBox("Rate", 0.0, 15.0, 5.0, step=0.01, decimals=3, suffix="%")
        self.rate_slider.valueChanged.connect(lambda _: self.params_changed.emit())
        self._layout.addWidget(self.rate_slider)
        self._layout.addSpacing(6)

        # --- Greek selector for surface ---
        self._layout.addWidget(_SectionLabel("3D SURFACE GREEK"))
        self.surface_greek_combo = QComboBox()
        for g in ["delta", "gamma", "theta", "vega", "rho"]:
            self.surface_greek_combo.addItem(g.capitalize(), g)
        self.surface_greek_combo.currentIndexChanged.connect(lambda _: self.params_changed.emit())
        self._layout.addWidget(self.surface_greek_combo)

        # --- Sensitivity mode ---
        self._layout.addWidget(_SectionLabel("SENSITIVITY MODE"))
        self.sensitivity_combo = QComboBox()
        self.sensitivity_combo.addItem("Delta vs IV", "delta_iv")
        self.sensitivity_combo.addItem("Gamma vs DTE", "gamma_dte")
        self.sensitivity_combo.addItem("Theta vs DTE", "theta_dte")
        self.sensitivity_combo.addItem("Vega vs IV", "vega_iv")
        self.sensitivity_combo.currentIndexChanged.connect(lambda _: self.params_changed.emit())
        self._layout.addWidget(self.sensitivity_combo)

        self._layout.addStretch()

        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # --- Slots ---

    def _on_fetch(self):
        ticker = self.ticker_input.text().strip().upper()
        if ticker:
            self.ticker_input.setText(ticker)
            self.fetch_requested.emit(ticker)

    def _on_strategy_changed(self, display_name: str):
        key = _STRATEGY_TEMPLATES.get(display_name)
        self.strategy_changed.emit(key)

    def _on_remove_leg(self):
        selected = self.legs_table.currentRow()
        if selected >= 0:
            self.remove_leg_requested.emit(selected)

    def _on_leg_widget_changed(self):
        """Any in-table combo/spinbox changed → emit legs_changed."""
        if not self._populating:
            self.legs_changed.emit()

    # --- Public helpers ---

    def set_status(self, text: str, color: str = Colors.TEXT_SECONDARY):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color}; font-size: {Fonts.SIZE_XS}px;")

    def set_spot_info(self, spot: float, rate: float, div: float):
        self.spot_label.setText(f"Spot: ${spot:,.2f}")
        self.rate_label.setText(f"r: {rate:.2%}")
        self.div_label.setText(f"q: {div:.2%}")

    def set_expirations(self, expirations: list[str]):
        self.expiry_combo.blockSignals(True)
        self.expiry_combo.clear()
        for exp in expirations:
            self.expiry_combo.addItem(exp)
        self.expiry_combo.blockSignals(False)
        if expirations:
            self.expiry_combo.setCurrentIndex(0)

    def set_dte_from_expiry(self, dte_days: float):
        """Set the DTE slider max and current value from actual DTE."""
        max_dte = max(dte_days, 1.0)
        self.dte_slider.set_maximum(max_dte)
        self.dte_slider.reset(dte_days)
        # Update Target max too, default to halfway
        self.target_dte_slider.set_maximum(max_dte)
        self.target_dte_slider.reset(dte_days / 2.0)

    def populate_legs_table(self, legs_data: list[dict]):
        """Fill the active legs table with editable widgets.

        Each row gets:
        - Col 0: QComboBox (Call/Put)
        - Col 1: QComboBox (Long/Short)
        - Col 2: QDoubleSpinBox (Strike)
        - Col 3: QSpinBox (Qty)
        - Col 4: QDoubleSpinBox (IV%)
        - Col 5: QTableWidgetItem (Expiry, read-only)
        """
        self._populating = True
        self.legs_table.setRowCount(0)

        for leg in legs_data:
            row = self.legs_table.rowCount()
            self.legs_table.insertRow(row)

            # Col 0: Type combo
            type_combo = QComboBox()
            type_combo.addItems(["call", "put"])
            type_combo.setCurrentText(leg.get("type", "call"))
            type_combo.setMinimumWidth(75)
            type_combo.currentIndexChanged.connect(self._on_leg_widget_changed)
            self.legs_table.setCellWidget(row, 0, type_combo)

            # Col 1: Dir combo
            dir_combo = QComboBox()
            dir_combo.addItems(["long", "short"])
            dir_combo.setCurrentText(leg.get("dir", "long"))
            dir_combo.setMinimumWidth(75)
            dir_combo.currentIndexChanged.connect(self._on_leg_widget_changed)
            self.legs_table.setCellWidget(row, 1, dir_combo)

            # Col 2: Strike spinbox
            strike_spin = QDoubleSpinBox()
            strike_spin.setRange(0.01, 99999.0)
            strike_spin.setDecimals(1)
            strike_spin.setSingleStep(1.0)
            strike_spin.setValue(leg.get("strike", 100.0))
            strike_spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
            strike_spin.valueChanged.connect(self._on_leg_widget_changed)
            self.legs_table.setCellWidget(row, 2, strike_spin)

            # Col 3: Qty spinbox
            qty_spin = QSpinBox()
            qty_spin.setRange(1, 9999)
            qty_spin.setValue(leg.get("qty", 1))
            qty_spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
            qty_spin.valueChanged.connect(self._on_leg_widget_changed)
            self.legs_table.setCellWidget(row, 3, qty_spin)

            # Col 4: IV spinbox (in %)
            iv_spin = QDoubleSpinBox()
            iv_spin.setRange(0.1, 500.0)
            iv_spin.setDecimals(1)
            iv_spin.setSingleStep(0.5)
            iv_spin.setSuffix("%")
            iv_spin.setValue(leg.get("iv", 0.20) * 100.0)
            iv_spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
            iv_spin.valueChanged.connect(self._on_leg_widget_changed)
            self.legs_table.setCellWidget(row, 4, iv_spin)

            # Col 5: Expiry (read-only text)
            expiry_raw = leg.get("expiry", "")
            try:
                exp_date = datetime.strptime(expiry_raw, "%Y-%m-%d")
                expiry_display = exp_date.strftime("%m-%d")
            except (ValueError, TypeError):
                expiry_display = expiry_raw[:10] if expiry_raw else "\u2014"
            expiry_item = QTableWidgetItem(expiry_display)
            expiry_item.setFlags(expiry_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            expiry_item.setForeground(QBrush(QColor(Colors.TEXT_SECONDARY)))
            self.legs_table.setItem(row, 5, expiry_item)

        self._populating = False

    def read_legs_from_table(self) -> list[dict]:
        """Read current leg data from the editable table widgets."""
        legs = []
        for row in range(self.legs_table.rowCount()):
            type_combo = self.legs_table.cellWidget(row, 0)
            dir_combo = self.legs_table.cellWidget(row, 1)
            strike_spin = self.legs_table.cellWidget(row, 2)
            qty_spin = self.legs_table.cellWidget(row, 3)
            iv_spin = self.legs_table.cellWidget(row, 4)
            expiry_item = self.legs_table.item(row, 5)

            if not all([type_combo, dir_combo, strike_spin, qty_spin, iv_spin]):
                continue

            legs.append({
                "type": type_combo.currentText(),
                "dir": dir_combo.currentText(),
                "strike": strike_spin.value(),
                "qty": qty_spin.value(),
                "iv": iv_spin.value() / 100.0,  # back to decimal
                "expiry": expiry_item.data(Qt.ItemDataRole.UserRole) or "",
            })
        return legs

    @property
    def iv_shift(self) -> float:
        """IV shift in decimal (e.g. slider=5% → 0.05)."""
        return self.iv_slider.value() / 100.0

    @property
    def dte_override(self) -> float:
        """DTE slider value in days."""
        return self.dte_slider.value()

    @property
    def target_dte_override(self) -> float:
        """Target days passed slider value."""
        return self.target_dte_slider.value()

    @property
    def rate_override(self) -> float:
        """Risk-free rate override (slider is in %, we return decimal)."""
        return self.rate_slider.value() / 100.0

    @property
    def surface_greek(self) -> str:
        return self.surface_greek_combo.currentData()

    @property
    def sensitivity_mode(self) -> str:
        return self.sensitivity_combo.currentData()
