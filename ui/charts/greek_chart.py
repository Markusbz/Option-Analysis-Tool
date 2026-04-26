"""
ui/charts/greek_chart.py

Greeks vs. Underlying Price — stacked subplot layout.
Three linked subplots:
  - Top:    Delta / Gamma
  - Middle: Theta / Vega
  - Bottom: Rho

All X-axes are linked so pan/zoom applies simultaneously.
Each subplot has its own Y-axis scale — preventing the flatlining issue.
Includes crosshair with interpolated value readout.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QVBoxLayout, QWidget, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from ui.theme import Colors, Fonts


# Subplot configuration: (subplot_index, greek_name, display, color, width)
_SUBPLOT_CONFIG = [
    # Subplot 0: Delta & Gamma
    (0, "delta", "Δ Delta", Colors.ACCENT_BLUE, 2.0),
    (0, "gamma", "Γ Gamma", Colors.ACCENT_GREEN, 1.8),
    # Subplot 1: Theta & Vega
    (1, "theta", "Θ Theta", Colors.ACCENT_ORANGE, 1.8),
    (1, "vega",  "ν Vega",  Colors.ACCENT_PURPLE, 1.8),
    # Subplot 2: Rho
    (2, "rho",   "ρ Rho",   Colors.ACCENT_CYAN, 1.5),
]

_SUBPLOT_TITLES = ["Delta / Gamma", "Theta / Vega", "Rho"]
_NUM_SUBPLOTS = 3


class GreekChart(QWidget):
    """Multi-subplot Greek chart with linked X-axes and crosshair readout."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {Colors.CHART_BG};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Crosshair readout label
        self._readout = QLabel("")
        self._readout.setFont(QFont(Fonts.MONO.split(",")[0].strip(), Fonts.SIZE_XS))
        self._readout.setStyleSheet(
            f"color: {Colors.TEXT_PRIMARY}; background: {Colors.BG_SURFACE}CC;"
            f"padding: 4px 8px; border-radius: 4px;"
        )
        self._readout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._readout.setFixedHeight(20)
        layout.addWidget(self._readout)

        # Graphics layout for subplots
        self._gw = pg.GraphicsLayoutWidget()
        self._gw.setBackground(Colors.CHART_BG)
        layout.addWidget(self._gw, stretch=1)

        self._plots: list[pg.PlotItem] = []
        self._curves: dict[str, pg.PlotDataItem] = {}
        self._crosshairs: list[pg.InfiniteLine] = []
        self._spot_lines: list[pg.InfiniteLine] = []
        self._data: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # name → (x, y)

        self._build_subplots()

    def _build_subplots(self):
        for i in range(_NUM_SUBPLOTS):
            plot = self._gw.addPlot(row=i, col=0)
            plot.hideButtons()
            for spine in ("top", "right"):
                plot.showAxis(spine, False)

            # Axis styling
            for axis_name in ("bottom", "left"):
                axis = plot.getAxis(axis_name)
                axis.setPen(pg.mkPen(Colors.CHART_AXIS, width=1))
                axis.setTextPen(pg.mkPen(Colors.CHART_AXIS))
                axis.setStyle(tickLength=-6)

            # Only show x-axis label on bottom subplot
            if i == _NUM_SUBPLOTS - 1:
                plot.setLabel("bottom", "Underlying Price", color=Colors.TEXT_SECONDARY)
            else:
                plot.showAxis("bottom", False)

            plot.setLabel("left", _SUBPLOT_TITLES[i], color=Colors.TEXT_SECONDARY,
                         **{"font-size": "9px"})
            plot.showGrid(x=True, y=True, alpha=0.12)

            # Zero line
            zero = pg.InfiniteLine(
                pos=0, angle=0,
                pen=pg.mkPen(Colors.BORDER, width=1,
                             style=pg.QtCore.Qt.PenStyle.DashLine),
            )
            plot.addItem(zero)

            # Crosshair (vertical line)
            vline = pg.InfiniteLine(
                angle=90,
                pen=pg.mkPen(Colors.CHART_CROSSHAIR, width=1,
                             style=pg.QtCore.Qt.PenStyle.DotLine),
            )
            vline.setVisible(False)
            plot.addItem(vline, ignoreBounds=True)
            self._crosshairs.append(vline)

            # Legend — anchored top-right
            plot.addLegend(
                offset=(-10, 5),
                brush=pg.mkBrush(Colors.BG_SURFACE + "CC"),
                pen=pg.mkPen(Colors.BORDER),
                labelTextColor=Colors.TEXT_PRIMARY,
            )

            self._plots.append(plot)

        # Link X-axes
        for i in range(1, _NUM_SUBPLOTS):
            self._plots[i].setXLink(self._plots[0])

        # Mouse tracking on the first plot (broadcasts to all via linked axes)
        for plot in self._plots:
            plot.scene().sigMouseMoved.connect(self._on_mouse_moved)

    def _on_mouse_moved(self, pos):
        """Track mouse across any subplot and update all crosshairs + readout."""
        # Find which plot the mouse is in
        for plot in self._plots:
            vb = plot.vb
            if vb.sceneBoundingRect().contains(pos):
                mouse_point = vb.mapSceneToView(pos)
                x_val = mouse_point.x()

                # Update all crosshairs
                for vline in self._crosshairs:
                    vline.setPos(x_val)
                    vline.setVisible(True)

                # Interpolate values
                parts = [f"S={x_val:.1f}"]
                for name, (x_data, y_data) in self._data.items():
                    if len(x_data) > 1:
                        y_interp = np.interp(x_val, x_data, y_data)
                        # Find the color for this greek
                        color = Colors.TEXT_PRIMARY
                        for _, gname, _, gcolor, _ in _SUBPLOT_CONFIG:
                            if gname == name:
                                color = gcolor
                                break
                        symbol = name[0].upper()
                        parts.append(
                            f'<span style="color:{color}">{symbol}={y_interp:+.2f}</span>'
                        )
                self._readout.setText("  ".join(parts))
                return

        # Mouse outside all plots
        for vline in self._crosshairs:
            vline.setVisible(False)
        self._readout.setText("")

    def update_plot(self, spot_range: np.ndarray,
                    greeks: dict[str, np.ndarray],
                    spot_price: float = None):
        """Redraw all Greek subplots.

        Parameters
        ----------
        spot_range : array
            X-axis (underlying prices).
        greeks : dict
            Mapping of greek_name → values array.
        spot_price : float, optional
            Current spot for vertical marker.
        """
        # Clear old curves
        for curve in self._curves.values():
            for plot in self._plots:
                try:
                    plot.removeItem(curve)
                except Exception:
                    pass
        self._curves.clear()
        self._data.clear()

        # Clear old legends
        for plot in self._plots:
            legend = plot.legend
            if legend is not None:
                legend.clear()

        # Remove old spot lines
        for line in self._spot_lines:
            for plot in self._plots:
                try:
                    plot.removeItem(line)
                except Exception:
                    pass
        self._spot_lines.clear()

        # Draw curves
        for subplot_idx, name, display, color, width in _SUBPLOT_CONFIG:
            if name not in greeks:
                continue
            values = greeks[name]
            plot = self._plots[subplot_idx]
            pen = pg.mkPen(color, width=width)
            curve = plot.plot(spot_range, values, pen=pen, name=display)
            self._curves[name] = curve
            self._data[name] = (spot_range, values)

        # Spot price marker on all subplots
        if spot_price is not None:
            for plot in self._plots:
                line = pg.InfiniteLine(
                    pos=spot_price, angle=90,
                    pen=pg.mkPen(Colors.ACCENT_CYAN, width=1,
                                 style=pg.QtCore.Qt.PenStyle.DashDotLine),
                )
                plot.addItem(line)
                self._spot_lines.append(line)
