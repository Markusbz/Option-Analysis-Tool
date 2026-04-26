"""
ui/charts/pnl_chart.py

PnL Payoff Diagram — pyqtgraph PlotWidget.
Plots T+0, T+N (halfway), and At-Expiration PnL curves
with green/red fill between for profit/loss regions.
Includes crosshair with interpolated value readout.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtGui import QFont

from ui.theme import Colors, Fonts


class PnLChart(pg.PlotWidget):
    """Interactive PnL payoff diagram with profit/loss fill regions and crosshair."""

    def __init__(self, parent=None):
        super().__init__(parent, background=Colors.CHART_BG)
        self._setup_axes()
        self._curves: dict[str, pg.PlotDataItem] = {}
        self._fills: list = []
        self._curve_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _setup_axes(self):
        """Configure axis styling to match the TradingView aesthetic."""
        plot = self.getPlotItem()

        # Remove the default bounding box
        plot.hideButtons()
        for spine in ("top", "right"):
            plot.showAxis(spine, False)

        # Style bottom / left axes
        for axis_name in ("bottom", "left"):
            axis = plot.getAxis(axis_name)
            axis.setPen(pg.mkPen(Colors.CHART_AXIS, width=1))
            axis.setTextPen(pg.mkPen(Colors.CHART_AXIS))
            axis.setStyle(tickLength=-8)

        # Grid lines — low-opacity dashed
        plot.showGrid(x=True, y=True, alpha=0.12)

        # Labels
        plot.setLabel("bottom", "Underlying Price", color=Colors.TEXT_SECONDARY)
        plot.setLabel("left", "P&L ($)", color=Colors.TEXT_SECONDARY)

        # Zero line
        self._zero_line = pg.InfiniteLine(
            pos=0, angle=0,
            pen=pg.mkPen(Colors.BORDER, width=1, style=pg.QtCore.Qt.PenStyle.DashLine),
        )
        plot.addItem(self._zero_line)

        # Crosshair
        self._vline = pg.InfiniteLine(
            angle=90,
            pen=pg.mkPen(Colors.CHART_CROSSHAIR, width=1,
                         style=pg.QtCore.Qt.PenStyle.DotLine),
        )
        self._vline.setVisible(False)
        plot.addItem(self._vline, ignoreBounds=True)

        # Value readout TextItem (top-left corner)
        self._readout = pg.TextItem(
            text="", anchor=(0, 0),
            color=Colors.TEXT_PRIMARY,
            fill=pg.mkBrush(Colors.BG_SURFACE + "DD"),
        )
        self._readout.setFont(QFont(Fonts.MONO.split(",")[0].strip(), Fonts.SIZE_XS))
        self._readout.setVisible(False)
        plot.addItem(self._readout, ignoreBounds=True)

        # Mouse tracking for crosshair
        self.scene().sigMouseMoved.connect(self._on_mouse_moved)

        # Legend — anchored bottom-right to avoid blocking data/readout
        self._legend = plot.addLegend(
            offset=(-10, -10),
            brush=pg.mkBrush(Colors.BG_SURFACE + "CC"),
            pen=pg.mkPen(Colors.BORDER),
            labelTextColor=Colors.TEXT_PRIMARY,
        )

    def _on_mouse_moved(self, pos):
        plot = self.getPlotItem()
        vb = plot.vb
        if vb.sceneBoundingRect().contains(pos):
            mouse_point = vb.mapSceneToView(pos)
            x_val = mouse_point.x()
            self._vline.setPos(x_val)
            self._vline.setVisible(True)

            # Interpolate Y values and build readout
            lines = [f"S = ${x_val:.1f}"]
            colors_map = {
                "expiry": Colors.TEXT_PRIMARY,
                "t0": Colors.ACCENT_BLUE,
                "mid": Colors.ACCENT_ORANGE,
            }
            names_map = {
                "expiry": "Expiry",
                "t0": "T+0",
                "mid": getattr(self, "_target_label", "Target"),
            }
            for key, (x_data, y_data) in self._curve_data.items():
                if len(x_data) > 1:
                    y_interp = float(np.interp(x_val, x_data, y_data))
                    display = names_map.get(key, key)
                    lines.append(f"{display}: ${y_interp:+,.0f}")

            self._readout.setText("\n".join(lines))
            self._readout.setVisible(True)

            # Position readout in top-left of view
            view_range = vb.viewRange()
            self._readout.setPos(view_range[0][0], view_range[1][1])
        else:
            self._vline.setVisible(False)
            self._readout.setVisible(False)

    def update_plot(self, spot_range: np.ndarray, pnl_expiry: np.ndarray,
                    pnl_t0: np.ndarray = None, pnl_mid: np.ndarray = None,
                    spot_price: float = None, target_label: str = "Target"):
        """Redraw the PnL curves and fill regions.

        Parameters
        ----------
        spot_range : array
            X-axis values (underlying prices).
        pnl_expiry : array
            PnL at expiration.
        pnl_t0 : array, optional
            PnL today (T+0).
        pnl_mid : array, optional
            PnL at midpoint to expiration (T+N).
        spot_price : float, optional
            Current spot price (draws a vertical marker).
        """
        plot = self.getPlotItem()
        self._target_label = target_label

        # Clear previous
        for item in self._fills:
            plot.removeItem(item)
        self._fills.clear()
        for name, curve in list(self._curves.items()):
            plot.removeItem(curve)
        self._curves.clear()
        self._curve_data.clear()
        self._legend.clear()

        # At-Expiration curve (main)
        pen_exp = pg.mkPen(Colors.TEXT_PRIMARY, width=2)
        self._curves["expiry"] = plot.plot(
            spot_range, pnl_expiry, pen=pen_exp, name="At Expiry"
        )
        self._curve_data["expiry"] = (spot_range, pnl_expiry)

        # Fill profit/loss regions under expiry curve
        zero = np.zeros_like(pnl_expiry)
        profit_y = np.where(pnl_expiry > 0, pnl_expiry, 0)
        loss_y = np.where(pnl_expiry < 0, pnl_expiry, 0)

        profit_curve = pg.PlotDataItem(spot_range, profit_y)
        zero_curve_p = pg.PlotDataItem(spot_range, zero)
        fill_profit = pg.FillBetweenItem(profit_curve, zero_curve_p,
                                         brush=pg.mkBrush(Colors.FILL_PROFIT))
        plot.addItem(fill_profit)
        self._fills.append(fill_profit)

        loss_curve = pg.PlotDataItem(spot_range, loss_y)
        zero_curve_l = pg.PlotDataItem(spot_range, zero)
        fill_loss = pg.FillBetweenItem(zero_curve_l, loss_curve,
                                       brush=pg.mkBrush(Colors.FILL_LOSS))
        plot.addItem(fill_loss)
        self._fills.append(fill_loss)

        # T+0 curve
        if pnl_t0 is not None:
            pen_t0 = pg.mkPen(Colors.ACCENT_BLUE, width=2)
            self._curves["t0"] = plot.plot(
                spot_range, pnl_t0, pen=pen_t0, name="T+0 (Today)"
            )
            self._curve_data["t0"] = (spot_range, pnl_t0)

        # T+N curve
        if pnl_mid is not None:
            pen_mid = pg.mkPen(Colors.ACCENT_ORANGE, width=1.5,
                               style=pg.QtCore.Qt.PenStyle.DashLine)
            self._curves["mid"] = plot.plot(
                spot_range, pnl_mid, pen=pen_mid, name=target_label
            )
            self._curve_data["mid"] = (spot_range, pnl_mid)

        # Spot price marker
        if spot_price is not None:
            if hasattr(self, "_spot_line"):
                plot.removeItem(self._spot_line)
            self._spot_line = pg.InfiniteLine(
                pos=spot_price, angle=90,
                pen=pg.mkPen(Colors.ACCENT_CYAN, width=1,
                             style=pg.QtCore.Qt.PenStyle.DashDotLine),
                label=f"Spot ${spot_price:.2f}",
                labelOpts={
                    "position": 0.95,
                    "color": Colors.ACCENT_CYAN,
                    "fill": pg.mkBrush(Colors.BG_SURFACE + "CC"),
                },
            )
            plot.addItem(self._spot_line)
