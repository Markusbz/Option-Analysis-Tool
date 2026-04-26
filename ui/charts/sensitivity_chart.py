"""
ui/charts/sensitivity_chart.py

Greek Sensitivity vs Volatility or Time — pyqtgraph PlotWidget.
Shows how a selected Greek varies as IV or DTE changes,
for a fixed spot price (or range of spot prices as multiple curves).
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from ui.theme import Colors

# Colour palette for multiple sensitivity curves
_CURVE_COLORS = [
    Colors.ACCENT_BLUE,
    Colors.ACCENT_GREEN,
    Colors.ACCENT_ORANGE,
    Colors.ACCENT_PURPLE,
    Colors.ACCENT_CYAN,
    Colors.ACCENT_RED,
    "#FFD54F",   # amber
    "#80CBC4",   # teal light
]


class SensitivityChart(pg.PlotWidget):
    """Greek sensitivity plot (e.g. Delta vs IV at multiple spot levels)."""

    def __init__(self, parent=None):
        super().__init__(parent, background=Colors.CHART_BG)
        self._setup_axes()
        self._curves = []

    def _setup_axes(self):
        plot = self.getPlotItem()
        plot.hideButtons()
        for spine in ("top", "right"):
            plot.showAxis(spine, False)

        for axis_name in ("bottom", "left"):
            axis = plot.getAxis(axis_name)
            axis.setPen(pg.mkPen(Colors.CHART_AXIS, width=1))
            axis.setTextPen(pg.mkPen(Colors.CHART_AXIS))
            axis.setStyle(tickLength=-8)

        plot.showGrid(x=True, y=True, alpha=0.12)
        plot.setLabel("bottom", "IV Shift (%)", color=Colors.TEXT_SECONDARY)
        plot.setLabel("left", "Greek Value", color=Colors.TEXT_SECONDARY)

        self._zero_line = pg.InfiniteLine(
            pos=0, angle=0,
            pen=pg.mkPen(Colors.BORDER, width=1,
                         style=pg.QtCore.Qt.PenStyle.DashLine),
        )
        plot.addItem(self._zero_line)

        self._legend = plot.addLegend(
            offset=(10, 10),
            brush=pg.mkBrush(Colors.BG_SURFACE + "CC"),
            pen=pg.mkPen(Colors.BORDER),
            labelTextColor=Colors.TEXT_PRIMARY,
        )

    def update_plot(
        self,
        x_values: np.ndarray,
        curves: dict[str, np.ndarray],
        x_label: str = "IV Shift (%)",
        y_label: str = "Greek Value",
    ):
        """Redraw sensitivity curves.

        Parameters
        ----------
        x_values : array
            Shared x-axis values (e.g. IV shifts or DTE values).
        curves : dict
            Mapping of curve_label → y-values.
        x_label, y_label : str
            Axis labels.
        """
        plot = self.getPlotItem()

        for curve in self._curves:
            plot.removeItem(curve)
        self._curves.clear()
        self._legend.clear()

        plot.setLabel("bottom", x_label, color=Colors.TEXT_SECONDARY)
        plot.setLabel("left", y_label, color=Colors.TEXT_SECONDARY)

        for idx, (label, y_vals) in enumerate(curves.items()):
            color = _CURVE_COLORS[idx % len(_CURVE_COLORS)]
            pen = pg.mkPen(color, width=2)
            c = plot.plot(x_values, y_vals, pen=pen, name=label)
            self._curves.append(c)
