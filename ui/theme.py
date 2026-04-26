"""
ui/theme.py

TradingView-inspired dark theme configuration.
Defines the color palette, font families, and QSS stylesheet
for a premium "Apple meets TradingView" look.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Color Palette  (TradingView / dark-pro inspired)
# ---------------------------------------------------------------------------

class Colors:
    """Centralised color constants used across all UI components."""

    # Backgrounds
    BG_PRIMARY = "#131722"       # Main window background (TradingView dark)
    BG_SURFACE = "#1E222D"       # Cards, panels, chart backgrounds
    BG_ELEVATED = "#252A37"      # Hovered elements, tooltips
    BG_INPUT = "#1A1E2B"         # Input field backgrounds

    # Borders
    BORDER = "#2A2E39"           # Subtle separator lines
    BORDER_FOCUS = "#2962FF"     # Focused input border

    # Text
    TEXT_PRIMARY = "#D1D4DC"     # Main text
    TEXT_SECONDARY = "#787B86"   # Labels, captions, muted text
    TEXT_DISABLED = "#434651"    # Disabled elements

    # Accents — trading semantics
    ACCENT_BLUE = "#2962FF"      # Primary action, call options
    ACCENT_GREEN = "#089981"     # Long, profit, bullish
    ACCENT_RED = "#F23645"       # Short, loss, bearish
    ACCENT_ORANGE = "#FF9800"    # Warnings, theta
    ACCENT_PURPLE = "#AB47BC"    # Vega / volatility
    ACCENT_CYAN = "#26C6DA"      # Rho / rate sensitivity

    # Chart-specific
    CHART_BG = "#131722"
    CHART_GRID = "#1E222D"
    CHART_AXIS = "#787B86"
    CHART_CROSSHAIR = "#9598A1"

    # PnL fill
    FILL_PROFIT = "#08998140"    # Green with ~25% opacity
    FILL_LOSS = "#F2364540"      # Red with ~25% opacity


# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------

class Fonts:
    """Font family stacks — the first available font is used."""

    # UI elements: labels, buttons, menus
    UI = "Inter, Segoe UI, Roboto, Helvetica Neue, sans-serif"

    # Numerical / tabular data: prices, Greeks, tables
    MONO = "JetBrains Mono, Fira Code, Consolas, Courier New, monospace"

    # Sizes
    SIZE_XS = 11
    SIZE_SM = 12
    SIZE_MD = 14
    SIZE_LG = 16
    SIZE_XL = 18
    SIZE_TITLE = 22


# ---------------------------------------------------------------------------
# Global QSS Stylesheet
# ---------------------------------------------------------------------------

def build_stylesheet() -> str:
    """Return a comprehensive QSS stylesheet for the entire application."""
    c = Colors
    f = Fonts

    return f"""
    /* ===== Global ===== */
    QMainWindow, QWidget {{
        background-color: {c.BG_PRIMARY};
        color: {c.TEXT_PRIMARY};
        font-family: {f.UI};
        font-size: {f.SIZE_SM}px;
    }}

    /* ===== Scroll Bars ===== */
    QScrollBar:vertical {{
        background: {c.BG_PRIMARY};
        width: 8px;
        border-radius: 4px;
    }}
    QScrollBar::handle:vertical {{
        background: {c.BORDER};
        min-height: 30px;
        border-radius: 4px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {c.TEXT_SECONDARY};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    QScrollBar:horizontal {{
        background: {c.BG_PRIMARY};
        height: 8px;
        border-radius: 4px;
    }}
    QScrollBar::handle:horizontal {{
        background: {c.BORDER};
        min-width: 30px;
        border-radius: 4px;
    }}

    /* ===== Labels ===== */
    QLabel {{
        color: {c.TEXT_PRIMARY};
        border: none;
        background: transparent;
    }}
    QLabel[class="section-title"] {{
        color: {c.TEXT_SECONDARY};
        font-size: {f.SIZE_XS}px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: 8px 0 4px 0;
    }}

    /* ===== Buttons ===== */
    QPushButton {{
        background-color: {c.ACCENT_BLUE};
        color: #FFFFFF;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 600;
        font-size: {f.SIZE_SM}px;
        min-height: 28px;
    }}
    QPushButton:hover {{
        background-color: #3D7AFF;
    }}
    QPushButton:pressed {{
        background-color: #1A4FCC;
    }}
    QPushButton:disabled {{
        background-color: {c.BG_ELEVATED};
        color: {c.TEXT_DISABLED};
    }}
    QPushButton[class="danger"] {{
        background-color: {c.ACCENT_RED};
    }}
    QPushButton[class="danger"]:hover {{
        background-color: #FF4D5E;
    }}
    QPushButton[class="secondary"] {{
        background-color: {c.BG_ELEVATED};
        color: {c.TEXT_PRIMARY};
        border: 1px solid {c.BORDER};
    }}
    QPushButton[class="secondary"]:hover {{
        background-color: #2F3545;
        border-color: {c.TEXT_SECONDARY};
    }}

    /* ===== Line Edits ===== */
    QLineEdit {{
        background-color: {c.BG_INPUT};
        color: {c.TEXT_PRIMARY};
        border: 1px solid {c.BORDER};
        border-radius: 6px;
        padding: 6px 10px;
        font-family: {f.MONO};
        font-size: {f.SIZE_SM}px;
        selection-background-color: {c.ACCENT_BLUE};
    }}
    QLineEdit:focus {{
        border-color: {c.ACCENT_BLUE};
    }}

    /* ===== Combo Boxes ===== */
    QComboBox {{
        background-color: {c.BG_INPUT};
        color: {c.TEXT_PRIMARY};
        border: 1px solid {c.BORDER};
        border-radius: 6px;
        padding: 6px 10px;
        font-size: {f.SIZE_SM}px;
        min-height: 24px;
    }}
    QComboBox:hover {{
        border-color: {c.TEXT_SECONDARY};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 24px;
    }}
    QComboBox::down-arrow {{
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 5px solid {c.TEXT_SECONDARY};
        margin-right: 8px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {c.BG_SURFACE};
        color: {c.TEXT_PRIMARY};
        border: 1px solid {c.BORDER};
        selection-background-color: {c.ACCENT_BLUE};
        outline: none;
    }}

    /* ===== Sliders ===== */
    QSlider::groove:horizontal {{
        background: {c.BG_ELEVATED};
        height: 4px;
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: {c.ACCENT_BLUE};
        width: 14px;
        height: 14px;
        margin: -5px 0;
        border-radius: 7px;
    }}
    QSlider::handle:horizontal:hover {{
        background: #3D7AFF;
    }}
    QSlider::sub-page:horizontal {{
        background: {c.ACCENT_BLUE};
        border-radius: 2px;
    }}

    /* ===== Tables ===== */
    QTableWidget {{
        background-color: {c.BG_SURFACE};
        color: {c.TEXT_PRIMARY};
        border: 1px solid {c.BORDER};
        border-radius: 6px;
        gridline-color: {c.BORDER};
        font-family: {f.MONO};
        font-size: {f.SIZE_XS}px;
        selection-background-color: {c.ACCENT_BLUE}30;
    }}
    QTableWidget::item {{
        padding: 4px 8px;
        border: none;
    }}
    QHeaderView::section {{
        background-color: {c.BG_ELEVATED};
        color: {c.TEXT_SECONDARY};
        border: none;
        border-bottom: 1px solid {c.BORDER};
        padding: 6px 8px;
        font-weight: 600;
        font-size: {f.SIZE_XS}px;
    }}

    /* ===== Splitters ===== */
    QSplitter::handle {{
        background: {c.BORDER};
    }}
    QSplitter::handle:horizontal {{
        width: 1px;
    }}
    QSplitter::handle:vertical {{
        height: 1px;
    }}

    /* ===== Status Bar ===== */
    QStatusBar {{
        background-color: {c.BG_SURFACE};
        color: {c.TEXT_SECONDARY};
        font-size: {f.SIZE_XS}px;
        border-top: 1px solid {c.BORDER};
    }}

    /* ===== Group Box ===== */
    QGroupBox {{
        background: transparent;
        border: none;
        margin-top: 16px;
        padding-top: 8px;
    }}
    QGroupBox::title {{
        color: {c.TEXT_SECONDARY};
        font-size: {f.SIZE_XS}px;
        font-weight: 600;
        subcontrol-origin: margin;
        left: 0px;
        padding: 0 4px;
    }}

    /* ===== Tooltips ===== */
    QToolTip {{
        background-color: {c.BG_ELEVATED};
        color: {c.TEXT_PRIMARY};
        border: 1px solid {c.BORDER};
        border-radius: 4px;
        padding: 4px 8px;
        font-size: {f.SIZE_XS}px;
    }}
    """
