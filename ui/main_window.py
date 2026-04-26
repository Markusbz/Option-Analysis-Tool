"""
ui/main_window.py

QMainWindow with:
- Left sidebar (ControlPanel)
- Center: vertical split — Option Chain (top) + 2×2 chart grid (bottom)
- Bottom: SummaryBar
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QLabel, QFrame,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from ui.theme import Colors, Fonts
from ui.control_panel import ControlPanel
from ui.option_chain_widget import OptionChainWidget
from ui.charts.pnl_chart import PnLChart
from ui.charts.greek_chart import GreekChart
from ui.charts.surface_chart import SurfaceChart
from ui.charts.sensitivity_chart import SensitivityChart


class SummaryBar(QFrame):
    """Bottom bar showing net Greeks, max profit/loss, and breakevens."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(56)
        self.setStyleSheet(
            f"background-color: {Colors.BG_SURFACE};"
            f"border-top: 1px solid {Colors.BORDER};"
        )
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(20, 0, 20, 0)
        self._layout.setSpacing(24)
        self._labels: dict[str, QLabel] = {}

        mono = QFont(Fonts.MONO.split(",")[0].strip(), Fonts.SIZE_SM)

        for key, display, color in [
            ("delta", "Δ Delta", Colors.ACCENT_BLUE),
            ("gamma", "Γ Gamma", Colors.ACCENT_GREEN),
            ("theta", "Θ Theta", Colors.ACCENT_ORANGE),
            ("vega", "ν Vega", Colors.ACCENT_PURPLE),
            ("rho", "ρ Rho", Colors.ACCENT_CYAN),
            ("max_profit", "Max Profit", Colors.ACCENT_GREEN),
            ("max_loss", "Max Loss", Colors.ACCENT_RED),
            ("breakevens", "Breakevens", Colors.TEXT_PRIMARY),
        ]:
            container = QWidget()
            vl = QVBoxLayout(container)
            vl.setContentsMargins(0, 4, 0, 4)
            vl.setSpacing(0)

            title = QLabel(display)
            title.setStyleSheet(
                f"color: {Colors.TEXT_SECONDARY}; font-size: 9px; font-weight: 600;"
            )
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)

            value = QLabel("—")
            value.setFont(mono)
            value.setStyleSheet(f"color: {color};")
            value.setAlignment(Qt.AlignmentFlag.AlignCenter)

            vl.addWidget(title)
            vl.addWidget(value)

            self._labels[key] = value
            self._layout.addWidget(container)

    def update_summary(self, greeks: dict, max_profit: float,
                       max_loss: float, breakevens: list[float]):
        for name in ("delta", "gamma", "theta", "vega", "rho"):
            val = greeks.get(name, 0.0)
            self._labels[name].setText(f"{val:+.2f}")

        self._labels["max_profit"].setText(f"${max_profit:+,.0f}")
        self._labels["max_loss"].setText(f"${max_loss:+,.0f}")

        if breakevens:
            be_str = " / ".join(f"${b:.1f}" for b in breakevens[:3])
        else:
            be_str = "—"
        self._labels["breakevens"].setText(be_str)

    def clear_summary(self):
        for lbl in self._labels.values():
            lbl.setText("—")


class MainWindow(QMainWindow):
    """Main application window with sidebar + chain + 2×2 chart grid + summary."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---- Main horizontal: sidebar | center ----
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setHandleWidth(1)

        # Sidebar
        self.control_panel = ControlPanel()
        main_splitter.addWidget(self.control_panel)

        # Center vertical: option chain (top) + chart grid (bottom)
        center_splitter = QSplitter(Qt.Orientation.Vertical)
        center_splitter.setHandleWidth(1)

        # Option chain
        self.option_chain = OptionChainWidget()
        center_splitter.addWidget(self.option_chain)

        # Chart grid (2×2 nested splitters)
        chart_grid = QSplitter(Qt.Orientation.Vertical)
        chart_grid.setHandleWidth(1)

        # Top row: PnL + Greeks
        top_row = QSplitter(Qt.Orientation.Horizontal)
        top_row.setHandleWidth(1)
        self.pnl_chart = PnLChart()
        self.greek_chart = GreekChart()
        top_row.addWidget(self.pnl_chart)
        top_row.addWidget(self.greek_chart)
        top_row.setSizes([500, 500])

        # Bottom row: 3D Surface + Sensitivity
        bottom_row = QSplitter(Qt.Orientation.Horizontal)
        bottom_row.setHandleWidth(1)
        self.surface_chart = SurfaceChart()
        self.sensitivity_chart = SensitivityChart()
        bottom_row.addWidget(self.surface_chart)
        bottom_row.addWidget(self.sensitivity_chart)
        bottom_row.setSizes([500, 500])

        chart_grid.addWidget(top_row)
        chart_grid.addWidget(bottom_row)
        chart_grid.setSizes([400, 400])

        center_splitter.addWidget(chart_grid)
        center_splitter.setSizes([250, 700])

        main_splitter.addWidget(center_splitter)
        main_splitter.setSizes([300, 1200])
        main_splitter.setCollapsible(0, False)

        root.addWidget(main_splitter, stretch=1)

        # ---- Bottom: summary bar ----
        self.summary_bar = SummaryBar()
        root.addWidget(self.summary_bar)
