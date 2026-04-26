"""
Options Analysis Tool — Entry Point

A PyQt6 desktop application for advanced options strategy analysis
and Greek visualization using the Black-Scholes-Merton model.

Launch:
    .venv\\Scripts\\python.exe main.py
"""

import sys
import os

# CRITICAL: QtWebEngineWidgets must be imported BEFORE QApplication is created
from PyQt6.QtWebEngineWidgets import QWebEngineView  # noqa: F401

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QFontDatabase

import pyqtgraph as pg


def main():
    """Initialize and launch the Options Analysis Tool."""

    # High-DPI
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)

    # ---- PyQtGraph global config (MUST come before any chart creation) ----
    pg.setConfigOptions(antialias=True, background=None)

    # ---- Theme ----
    from ui.theme import build_stylesheet, Colors, Fonts
    app.setStyleSheet(build_stylesheet())

    # Set default font
    font = QFont("Inter", 11)
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    app.setFont(font)

    # ---- Build UI ----
    from ui.main_window import MainWindow
    from controller.app_controller import AppController

    window = MainWindow()
    window.setWindowTitle("Options Analysis Tool")
    window.resize(1680, 960)

    # ---- Wire controller ----
    controller = AppController(
        control_panel=window.control_panel,
        option_chain=window.option_chain,
        pnl_chart=window.pnl_chart,
        greek_chart=window.greek_chart,
        surface_chart=window.surface_chart,
        sensitivity_chart=window.sensitivity_chart,
        summary_bar=window.summary_bar,
    )

    # Keep a reference so it doesn't get garbage-collected
    window._controller = controller

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
