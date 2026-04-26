"""
ui/charts/surface_chart.py

3D Greek Surface / Heatmap — Plotly rendered in QWebEngineView.
Shows how a selected Greek changes across Spot × DTE (or Spot × IV).
Uses a debounced update to avoid choking the web engine.
"""

from __future__ import annotations

import json

import numpy as np
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QUrl

import plotly.graph_objects as go
from plotly.io import to_html

from ui.theme import Colors


class SurfaceChart(QWebEngineView):
    """3D Greek surface rendered via Plotly inside a QWebEngineView."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {Colors.CHART_BG}; border: none;")
        # Show a blank dark page initially
        self._show_placeholder()

    def _show_placeholder(self):
        html = f"""
        <html>
        <body style="background:{Colors.CHART_BG}; margin:0;
                     display:flex; align-items:center; justify-content:center;
                     height:100vh;">
            <p style="color:{Colors.TEXT_SECONDARY}; font-family:Inter,sans-serif;
                      font-size:14px;">
                3D Greek Surface — load data to visualise
            </p>
        </body>
        </html>
        """
        self.setHtml(html)

    def update_surface(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        x_label: str = "Underlying Price",
        y_label: str = "DTE (days)",
        z_label: str = "Delta",
    ):
        """Render a 3D surface plot.

        Parameters
        ----------
        x, y : 1-D arrays
            Axis tick values.
        z : 2-D array
            Surface values, shape (len(y), len(x)).
        x_label, y_label, z_label : str
            Axis labels.
        """
        fig = go.Figure(data=[
            go.Surface(
                x=x, y=y, z=z,
                colorscale="Plasma",
                colorbar=dict(
                    title=dict(text=z_label, font=dict(color=Colors.TEXT_SECONDARY, size=11)),
                    tickfont=dict(color=Colors.TEXT_SECONDARY, size=10),
                    bgcolor=Colors.BG_SURFACE,
                    bordercolor=Colors.BORDER,
                    borderwidth=1,
                    len=0.7,
                ),
                lighting=dict(ambient=0.6, diffuse=0.5, specular=0.15),
                hovertemplate=(
                    f"<b>{x_label}</b>: %{{x:.1f}}<br>"
                    f"<b>{y_label}</b>: %{{y:.1f}}<br>"
                    f"<b>{z_label}</b>: %{{z:.4f}}<extra></extra>"
                ),
            )
        ])

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
            scene=dict(
                xaxis=dict(
                    title=dict(text=x_label, font=dict(color=Colors.TEXT_SECONDARY, size=11)),
                    backgroundcolor=Colors.CHART_BG,
                    gridcolor=Colors.BORDER,
                    showbackground=True,
                    tickfont=dict(color=Colors.TEXT_SECONDARY, size=9),
                ),
                yaxis=dict(
                    title=dict(text=y_label, font=dict(color=Colors.TEXT_SECONDARY, size=11)),
                    backgroundcolor=Colors.CHART_BG,
                    gridcolor=Colors.BORDER,
                    showbackground=True,
                    tickfont=dict(color=Colors.TEXT_SECONDARY, size=9),
                ),
                zaxis=dict(
                    title=dict(text=z_label, font=dict(color=Colors.TEXT_SECONDARY, size=11)),
                    backgroundcolor=Colors.BG_SURFACE,
                    gridcolor=Colors.BORDER,
                    showbackground=True,
                    tickfont=dict(color=Colors.TEXT_SECONDARY, size=9),
                ),
                bgcolor=Colors.CHART_BG,
            ),
            font=dict(family="Inter, Segoe UI, sans-serif"),
        )

        html = to_html(
            fig,
            include_plotlyjs="cdn",
            full_html=True,
            config={
                "displayModeBar": True,
                "modeBarButtonsToRemove": ["toImage", "sendDataToCloud"],
                "displaylogo": False,
            },
        )

        # Inject background color into <body>
        html = html.replace(
            "<body>",
            f'<body style="background:{Colors.CHART_BG}; margin:0; overflow:hidden;">',
        )

        self.setHtml(html)
