"""
Welcome Screen - PyQt6 welcome window for CALQA
Displays options to create new project or open existing project
"""

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from view.themes import apply_theme


class WelcomeScreen(QMainWindow):
    """
    Welcome screen with two main action buttons.
    Signals allow controller to handle button clicks.
    """

    # Signals
    create_project_clicked = pyqtSignal()
    open_project_clicked = pyqtSignal()

    def __init__(self, theme: str = "light"):
        super().__init__()
        self.current_theme = theme
        self.init_ui()
        apply_theme(self, theme)

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("CALQA – Centella Asiatica Leaves Quality Assessment")
        self.setGeometry(100, 100, 800, 600)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(40, 40, 40, 40)

        # Title
        title_label = QLabel("CALQA")
        title_font = QFont()
        title_font.setPointSize(32)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)

        # Subtitle
        subtitle_label = QLabel(
            "Centella Asiatica Leaves Quality Assessment\nGPU-Accelerated Vein Segmentation"
        )
        subtitle_font = QFont()
        subtitle_font.setPointSize(12)
        subtitle_label.setFont(subtitle_font)
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(subtitle_label)

        # Spacer
        main_layout.addSpacing(40)

        # Button layout
        button_layout = QHBoxLayout()
        button_layout.setSpacing(20)

        # Create New Project button
        create_btn = QPushButton("Create New Project")
        create_btn.setFixedSize(200, 80)
        create_btn.setFont(self._get_button_font())
        create_btn.clicked.connect(self.on_create_clicked)
        button_layout.addWidget(create_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        # Open Existing Project button
        open_btn = QPushButton("Open Existing Project")
        open_btn.setFixedSize(200, 80)
        open_btn.setFont(self._get_button_font())
        open_btn.clicked.connect(self.on_open_clicked)
        button_layout.addWidget(open_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        main_layout.addLayout(button_layout)

        # Add stretch at the bottom
        main_layout.addStretch()

        central_widget.setLayout(main_layout)

    def on_create_clicked(self):
        """Emit signal when Create Project button is clicked."""
        self.create_project_clicked.emit()

    def on_open_clicked(self):
        """Emit signal when Open Project button is clicked."""
        self.open_project_clicked.emit()

    def _get_button_font(self) -> QFont:
        """Get font for buttons."""
        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        return font

    def set_theme(self, theme: str):
        """
        Switch theme at runtime.

        Args:
            theme: "light" or "dark"
        """
        self.current_theme = theme
        apply_theme(self, theme)
