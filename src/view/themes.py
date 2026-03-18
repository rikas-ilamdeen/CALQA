"""
Theme Manager - Provides light and dark theme stylesheets for PyQt6
CALQA uses purple accents (#8B5CF6) for University of Westminster branding
"""

from typing import Optional

# Purple accent color for CALQA branding
PRIMARY_COLOR = "#8B5CF6"
PRIMARY_DARK = "#6B21A8"
PRIMARY_LIGHT = "#A78BFA"

LIGHT_THEME = """
    /* Light Theme - CALQA */
    QMainWindow {
        background-color: #FFFFFF;
    }
    
    QWidget {
        background-color: #FFFFFF;
        color: #1F2937;
    }
    
    QMenuBar {
        background-color: #F3F4F6;
        color: #1F2937;
        border-bottom: 1px solid #E5E7EB;
    }
    
    QMenuBar::item:selected {
        background-color: """ + PRIMARY_LIGHT + """;
    }
    
    QMenu {
        background-color: #FFFFFF;
        color: #1F2937;
        border: 1px solid #E5E7EB;
    }
    
    QMenu::item:selected {
        background-color: """ + PRIMARY_LIGHT + """;
    }
    
    QPushButton {
        background-color: """ + PRIMARY_COLOR + """;
        color: #FFFFFF;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: bold;
        font-size: 13px;
    }
    
    QPushButton:hover {
        background-color: """ + PRIMARY_DARK + """;
    }
    
    QPushButton:pressed {
        background-color: #5B21B6;
    }
    
    QPushButton:disabled {
        background-color: #D1D5DB;
        color: #9CA3AF;
    }
    
    QLineEdit {
        background-color: #F9FAFB;
        color: #1F2937;
        border: 1px solid #E5E7EB;
        border-radius: 4px;
        padding: 6px 8px;
        font-size: 12px;
    }
    
    QLineEdit:focus {
        border: 2px solid """ + PRIMARY_COLOR + """;
    }
    
    QTextEdit {
        background-color: #F9FAFB;
        color: #1F2937;
        border: 1px solid #E5E7EB;
        border-radius: 4px;
        padding: 8px;
        font-size: 12px;
    }
    
    QTextEdit:focus {
        border: 2px solid """ + PRIMARY_COLOR + """;
    }
    
    QComboBox {
        background-color: #F9FAFB;
        color: #1F2937;
        border: 1px solid #E5E7EB;
        border-radius: 4px;
        padding: 6px 8px;
        font-size: 12px;
    }
    
    QComboBox:focus {
        border: 2px solid """ + PRIMARY_COLOR + """;
    }
    
    QComboBox::drop-down {
        border: none;
        width: 30px;
    }
    
    QComboBox QAbstractItemView {
        background-color: #FFFFFF;
        color: #1F2937;
        border: 1px solid #E5E7EB;
        selection-background-color: """ + PRIMARY_LIGHT + """;
    }
    
    QTabWidget::pane {
        border: 1px solid #E5E7EB;
    }
    
    QTabBar::tab {
        background-color: #F3F4F6;
        color: #1F2937;
        padding: 8px 16px;
        border: 1px solid #E5E7EB;
        border-bottom: none;
        margin-right: 2px;
    }
    
    QTabBar::tab:selected {
        background-color: #FFFFFF;
        color: """ + PRIMARY_COLOR + """;
        border-bottom: 3px solid """ + PRIMARY_COLOR + """;
    }
    
    QTabBar::tab:hover {
        background-color: #FFFBFE;
    }
    
    QLabel {
        color: #1F2937;
    }
    
    QRadioButton {
        color: #1F2937;
        spacing: 6px;
    }
    
    QRadioButton::indicator {
        width: 18px;
        height: 18px;
    }
    
    QRadioButton::indicator:unchecked {
        background-color: #F9FAFB;
        border: 2px solid #D1D5DB;
        border-radius: 9px;
    }
    
    QRadioButton::indicator:checked {
        background-color: """ + PRIMARY_COLOR + """;
        border: 2px solid """ + PRIMARY_COLOR + """;
        border-radius: 9px;
    }
    
    QProgressBar {
        background-color: #E5E7EB;
        border: 1px solid #D1D5DB;
        border-radius: 4px;
        padding: 3px;
        text-align: center;
        font-size: 11px;
    }
    
    QProgressBar::chunk {
        background-color: """ + PRIMARY_COLOR + """;
        border-radius: 3px;
    }
    
    QStatusBar {
        background-color: #F3F4F6;
        color: #1F2937;
        border-top: 1px solid #E5E7EB;
        font-size: 11px;
    }
    
    QTableWidget {
        background-color: #FFFFFF;
        alternate-background-color: #F9FAFB;
        gridline-color: #E5E7EB;
        border: 1px solid #E5E7EB;
    }
    
    QTableWidget::item:selected {
        background-color: """ + PRIMARY_LIGHT + """;
    }
    
    QHeaderView::section {
        background-color: #F3F4F6;
        color: #1F2937;
        padding: 6px 8px;
        border: none;
        border-right: 1px solid #E5E7EB;
        border-bottom: 1px solid #E5E7EB;
        font-weight: bold;
    }
    
    QScrollBar:vertical {
        background-color: #F9FAFB;
        width: 12px;
        border: none;
    }
    
    QScrollBar::handle:vertical {
        background-color: #D1D5DB;
        border-radius: 6px;
        min-height: 20px;
    }
    
    QScrollBar::handle:vertical:hover {
        background-color: #9CA3AF;
    }
    
    QScrollBar:horizontal {
        background-color: #F9FAFB;
        height: 12px;
        border: none;
    }
    
    QScrollBar::handle:horizontal {
        background-color: #D1D5DB;
        border-radius: 6px;
        min-width: 20px;
    }
    
    QScrollBar::handle:horizontal:hover {
        background-color: #9CA3AF;
    }
    
    QDialog {
        background-color: #FFFFFF;
    }
"""

DARK_THEME = """
    /* Dark Theme - CALQA */
    QMainWindow {
        background-color: #111827;
    }
    
    QWidget {
        background-color: #111827;
        color: #F3F4F6;
    }
    
    QMenuBar {
        background-color: #1F2937;
        color: #F3F4F6;
        border-bottom: 1px solid #374151;
    }
    
    QMenuBar::item:selected {
        background-color: """ + PRIMARY_COLOR + """;
    }
    
    QMenu {
        background-color: #1F2937;
        color: #F3F4F6;
        border: 1px solid #374151;
    }
    
    QMenu::item:selected {
        background-color: """ + PRIMARY_COLOR + """;
    }
    
    QPushButton {
        background-color: """ + PRIMARY_COLOR + """;
        color: #FFFFFF;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: bold;
        font-size: 13px;
    }
    
    QPushButton:hover {
        background-color: """ + PRIMARY_DARK + """;
    }
    
    QPushButton:pressed {
        background-color: #5B21B6;
    }
    
    QPushButton:disabled {
        background-color: #374151;
        color: #6B7280;
    }
    
    QLineEdit {
        background-color: #1F2937;
        color: #F3F4F6;
        border: 1px solid #374151;
        border-radius: 4px;
        padding: 6px 8px;
        font-size: 12px;
    }
    
    QLineEdit:focus {
        border: 2px solid """ + PRIMARY_COLOR + """;
    }
    
    QTextEdit {
        background-color: #1F2937;
        color: #F3F4F6;
        border: 1px solid #374151;
        border-radius: 4px;
        padding: 8px;
        font-size: 12px;
    }
    
    QTextEdit:focus {
        border: 2px solid """ + PRIMARY_COLOR + """;
    }
    
    QComboBox {
        background-color: #1F2937;
        color: #F3F4F6;
        border: 1px solid #374151;
        border-radius: 4px;
        padding: 6px 8px;
        font-size: 12px;
    }
    
    QComboBox:focus {
        border: 2px solid """ + PRIMARY_COLOR + """;
    }
    
    QComboBox::drop-down {
        border: none;
        width: 30px;
    }
    
    QComboBox QAbstractItemView {
        background-color: #1F2937;
        color: #F3F4F6;
        border: 1px solid #374151;
        selection-background-color: """ + PRIMARY_DARK + """;
    }
    
    QTabWidget::pane {
        border: 1px solid #374151;
    }
    
    QTabBar::tab {
        background-color: #1F2937;
        color: #9CA3AF;
        padding: 8px 16px;
        border: 1px solid #374151;
        border-bottom: none;
        margin-right: 2px;
    }
    
    QTabBar::tab:selected {
        background-color: #111827;
        color: """ + PRIMARY_LIGHT + """;
        border-bottom: 3px solid """ + PRIMARY_COLOR + """;
    }
    
    QTabBar::tab:hover {
        background-color: #2D3748;
    }
    
    QLabel {
        color: #F3F4F6;
    }
    
    QRadioButton {
        color: #F3F4F6;
        spacing: 6px;
    }
    
    QRadioButton::indicator {
        width: 18px;
        height: 18px;
    }
    
    QRadioButton::indicator:unchecked {
        background-color: #1F2937;
        border: 2px solid #4B5563;
        border-radius: 9px;
    }
    
    QRadioButton::indicator:checked {
        background-color: """ + PRIMARY_COLOR + """;
        border: 2px solid """ + PRIMARY_COLOR + """;
        border-radius: 9px;
    }
    
    QProgressBar {
        background-color: #1F2937;
        border: 1px solid #374151;
        border-radius: 4px;
        padding: 3px;
        text-align: center;
        font-size: 11px;
        color: #F3F4F6;
    }
    
    QProgressBar::chunk {
        background-color: """ + PRIMARY_COLOR + """;
        border-radius: 3px;
    }
    
    QStatusBar {
        background-color: #1F2937;
        color: #F3F4F6;
        border-top: 1px solid #374151;
        font-size: 11px;
    }
    
    QTableWidget {
        background-color: #111827;
        alternate-background-color: #1F2937;
        gridline-color: #374151;
        border: 1px solid #374151;
    }
    
    QTableWidget::item:selected {
        background-color: """ + PRIMARY_DARK + """;
    }
    
    QHeaderView::section {
        background-color: #1F2937;
        color: #F3F4F6;
        padding: 6px 8px;
        border: none;
        border-right: 1px solid #374151;
        border-bottom: 1px solid #374151;
        font-weight: bold;
    }
    
    QScrollBar:vertical {
        background-color: #111827;
        width: 12px;
        border: none;
    }
    
    QScrollBar::handle:vertical {
        background-color: #4B5563;
        border-radius: 6px;
        min-height: 20px;
    }
    
    QScrollBar::handle:vertical:hover {
        background-color: #6B7280;
    }
    
    QScrollBar:horizontal {
        background-color: #111827;
        height: 12px;
        border: none;
    }
    
    QScrollBar::handle:horizontal {
        background-color: #4B5563;
        border-radius: 6px;
        min-width: 20px;
    }
    
    QScrollBar::handle:horizontal:hover {
        background-color: #6B7280;
    }
    
    QDialog {
        background-color: #111827;
    }
"""


def apply_theme(widget, theme_name: str = "light") -> None:
    """
    Apply a theme stylesheet to a QWidget or QApplication.

    Args:
        widget: QWidget or QApplication instance
        theme_name: "light" or "dark"
    """
    if theme_name.lower() == "dark":
        widget.setStyleSheet(DARK_THEME)
    else:
        widget.setStyleSheet(LIGHT_THEME)


def get_theme_stylesheet(theme_name: str = "light") -> str:
    """
    Get theme stylesheet as string without applying.

    Args:
        theme_name: "light" or "dark"

    Returns:
        Stylesheet string
    """
    if theme_name.lower() == "dark":
        return DARK_THEME
    return LIGHT_THEME
