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
from PyQt6.QtCore import pyqtSignal
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
        
        # ViewBox constraints
        plot.vb.setMouseEnabled(x=True, y=False)
        plot.vb.enableAutoRange(axis=pg.ViewBox.YAxis)
        plot.vb.setAutoVisible(x=False, y=True)

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

        self._legend = plot.addLegend(
            offset=(-10, -10),
            brush=pg.mkBrush(Colors.BG_SURFACE + "CC"),
            pen=pg.mkPen(Colors.BORDER),
            labelTextColor=Colors.TEXT_PRIMARY,
        )

        # Initialize Curves
        self._curve_expiry = pg.PlotDataItem(pen=pg.mkPen(Colors.TEXT_PRIMARY, width=2), name="At Expiry")
        plot.addItem(self._curve_expiry)

        self._curve_t0 = pg.PlotDataItem(pen=pg.mkPen(Colors.ACCENT_BLUE, width=2), name="T+0 (Today)")
        plot.addItem(self._curve_t0)

        self._curve_mid = pg.PlotDataItem(pen=pg.mkPen(Colors.ACCENT_ORANGE, width=1.5, style=pg.QtCore.Qt.PenStyle.DashLine), name="Target")
        plot.addItem(self._curve_mid)

        # Initialize Fills
        self._curve_profit = pg.PlotDataItem()
        self._curve_profit_zero = pg.PlotDataItem()
        self._fill_profit = pg.FillBetweenItem(self._curve_profit, self._curve_profit_zero, brush=pg.mkBrush(Colors.FILL_PROFIT))
        plot.addItem(self._fill_profit, ignoreBounds=True)

        self._curve_loss = pg.PlotDataItem()
        self._curve_loss_zero = pg.PlotDataItem()
        self._fill_loss = pg.FillBetweenItem(self._curve_loss_zero, self._curve_loss, brush=pg.mkBrush(Colors.FILL_LOSS))
        plot.addItem(self._fill_loss, ignoreBounds=True)

        # Spot Price Marker
        self._spot_line = pg.InfiniteLine(
            angle=90,
            pen=pg.mkPen(Colors.ACCENT_CYAN, width=1, style=pg.QtCore.Qt.PenStyle.DashDotLine),
            label="Spot",
            labelOpts={"position": 0.95, "color": Colors.ACCENT_CYAN, "fill": pg.mkBrush(Colors.BG_SURFACE + "CC")},
        )
        self._spot_line.setVisible(False)
        plot.addItem(self._spot_line)

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

        # Update curves via setData
        self._curve_expiry.setData(spot_range, pnl_expiry)
        self._curve_data["expiry"] = (spot_range, pnl_expiry)

        # Update Fills
        zero = np.zeros_like(pnl_expiry)
        profit_y = np.where(pnl_expiry > 0, pnl_expiry, 0)
        loss_y = np.where(pnl_expiry < 0, pnl_expiry, 0)

        self._curve_profit.setData(spot_range, profit_y)
        self._curve_profit_zero.setData(spot_range, zero)
        
        self._curve_loss.setData(spot_range, loss_y)
        self._curve_loss_zero.setData(spot_range, zero)

        # Update T+0
        if pnl_t0 is not None:
            self._curve_t0.setData(spot_range, pnl_t0)
            self._curve_t0.setVisible(True)
            self._curve_data["t0"] = (spot_range, pnl_t0)
        else:
            self._curve_t0.setVisible(False)

        # Update T+N
        if pnl_mid is not None:
            self._curve_mid.setData(spot_range, pnl_mid)
            self._curve_mid.setVisible(True)
            self._curve_data["mid"] = (spot_range, pnl_mid)
            # Cannot easily rename a PlotDataItem in legend dynamically without recreate, 
            # but usually it's fine.
        else:
            self._curve_mid.setVisible(False)

        # Update Spot Line
        if spot_price is not None:
            self._spot_line.setPos(spot_price)
            self._spot_line.label.setFormat(f"Spot ${spot_price:.2f}")
            self._spot_line.setVisible(True)
        else:
            self._spot_line.setVisible(False)
