"""
ui/summary_panel.py

Vertical panel showing net Greeks, max profit/loss, and breakevens,
designed to sit alongside the option chain.
"""

from PyQt6.QtWidgets import QFrame, QVBoxLayout, QLabel, QGridLayout, QWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from ui.theme import Colors, Fonts


class SummaryPanel(QFrame):
    """Side panel displaying strategy summary metrics with large typography."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(280)
        self.setStyleSheet(
            f"background-color: {Colors.BG_SURFACE};"
            f"border-left: 1px solid {Colors.BORDER};"
        )
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(12, 8, 12, 8)
        self._layout.setSpacing(12)

        self._labels: dict[str, QLabel] = {}

        mono = QFont(Fonts.MONO.split(",")[0].strip(), 18, QFont.Weight.Bold)

        # Build Net Greeks Section
        self._layout.addWidget(self._build_section_header("NET GREEKS"))
        greeks_grid = QGridLayout()
        greeks_grid.setSpacing(6)
        
        greeks = [
            ("delta", "Δ Delta", Colors.ACCENT_BLUE),
            ("gamma", "Γ Gamma", Colors.ACCENT_GREEN),
            ("theta", "Θ Theta", Colors.ACCENT_ORANGE),
            ("vega", "ν Vega", Colors.ACCENT_PURPLE),
            ("rho", "ρ Rho", Colors.ACCENT_CYAN),
        ]
        for row, (key, display, color) in enumerate(greeks):
            title = QLabel(display)
            title.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 14px;")
            
            value = QLabel("—")
            value.setFont(mono)
            value.setStyleSheet(f"color: {color}; border: none;")
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            
            greeks_grid.addWidget(title, row, 0)
            greeks_grid.addWidget(value, row, 1)
            self._labels[key] = value

        greeks_widget = QWidget()
        greeks_widget.setStyleSheet("border: none;")
        greeks_widget.setLayout(greeks_grid)
        self._layout.addWidget(greeks_widget)

        # Build P&L Profile Section
        self._layout.addWidget(self._build_section_header("P&L PROFILE"))
        pnl_grid = QGridLayout()
        pnl_grid.setSpacing(6)
        
        pnl_metrics = [
            ("max_profit", "Max Profit", Colors.ACCENT_GREEN),
            ("max_loss", "Max Loss", Colors.ACCENT_RED),
            ("breakevens", "Breakevens", Colors.TEXT_PRIMARY),
        ]
        
        for row, (key, display, color) in enumerate(pnl_metrics):
            title = QLabel(display)
            title.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 14px;")
            
            value = QLabel("—")
            if key == "breakevens":
                value.setFont(QFont(Fonts.MONO.split(",")[0].strip(), 14, QFont.Weight.Bold))
            else:
                value.setFont(mono)
            value.setStyleSheet(f"color: {color}; border: none;")
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            
            pnl_grid.addWidget(title, row, 0)
            pnl_grid.addWidget(value, row, 1)
            self._labels[key] = value

        pnl_widget = QWidget()
        pnl_widget.setStyleSheet("border: none;")
        pnl_widget.setLayout(pnl_grid)
        self._layout.addWidget(pnl_widget)

        self._layout.addStretch(1)

    def _build_section_header(self, title: str) -> QLabel:
        lbl = QLabel(title)
        lbl.setStyleSheet(f"color: {Colors.TEXT_DISABLED}; font-size: 10px; font-weight: 800; border: none;")
        return lbl

    def update_summary(self, greeks: dict, max_profit: str,
                       max_loss: str, breakevens: list[float]):
        for name in ("delta", "gamma", "theta", "vega", "rho"):
            val = greeks.get(name, 0.0)
            self._labels[name].setText(f"{val:+.2f}")

        self._labels["max_profit"].setText(max_profit)
        self._labels["max_loss"].setText(max_loss)

        if breakevens:
            be_str = " / ".join(f"${b:.1f}" for b in breakevens[:3])
        else:
            be_str = "—"
        self._labels["breakevens"].setText(be_str)

    def clear_summary(self):
        for lbl in self._labels.values():
            lbl.setText("—")
