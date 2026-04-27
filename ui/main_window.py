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


from ui.summary_panel import SummaryPanel


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

        # Option chain + Summary Panel
        chain_row = QWidget()
        chain_row.setMaximumHeight(300)
        chain_layout = QHBoxLayout(chain_row)
        chain_layout.setContentsMargins(0, 0, 0, 0)
        chain_layout.setSpacing(0)
        
        self.option_chain = OptionChainWidget()
        self.summary_panel = SummaryPanel()
        
        chain_layout.addWidget(self.option_chain, stretch=1)
        chain_layout.addWidget(self.summary_panel)
        
        center_splitter.addWidget(chain_row)

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
        center_splitter.setSizes([150, 900])

        main_splitter.addWidget(center_splitter)
        main_splitter.setSizes([300, 1200])
        main_splitter.setCollapsible(0, False)

        root.addWidget(main_splitter, stretch=1)


