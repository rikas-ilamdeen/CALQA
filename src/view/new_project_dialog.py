"""
New Project Dialog - PyQt6 dialog for creating new CALQA projects
Collects project metadata and validates input
"""

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QTextEdit,
    QComboBox,
    QPushButton,
    QFileDialog,
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont
from pathlib import Path
from typing import Optional, Dict

from view.themes import apply_theme


class NewProjectDialog(QDialog):
    """
    Dialog for creating a new project.
    Returns project data dict on successful completion.
    """

    project_created = pyqtSignal(dict)

    def __init__(self, parent=None, theme: str = "light"):
        super().__init__(parent)
        self.current_theme = theme
        self.project_data: Optional[Dict] = None
        self.init_ui()
        apply_theme(self, theme)

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Create New Project – CALQA")
        self.setGeometry(150, 150, 600, 500)
        self.setModal(True)

        layout = QVBoxLayout()
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title_label = QLabel("New Project")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        # Project Name
        layout.addWidget(QLabel("Project Name *"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter project name (required)")
        layout.addWidget(self.name_input)

        # Description
        layout.addWidget(QLabel("Description"))
        self.description_input = QTextEdit()
        self.description_input.setPlaceholderText("Enter project description")
        self.description_input.setMaximumHeight(80)
        layout.addWidget(self.description_input)

        # Project Type
        layout.addWidget(QLabel("Project Type"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["SEGMENTATION", "CLASSIFICATION"])
        layout.addWidget(self.type_combo)

        # Project Folder
        layout.addWidget(QLabel("Project Folder *"))
        folder_layout = QHBoxLayout()
        self.folder_input = QLineEdit()
        self.folder_input.setPlaceholderText("Choose folder for project")
        self.folder_input.setReadOnly(True)
        folder_btn = QPushButton("Browse...")
        folder_btn.setMaximumWidth(100)
        folder_btn.clicked.connect(self.browse_folder)
        folder_layout.addWidget(self.folder_input)
        folder_layout.addWidget(folder_btn)
        layout.addLayout(folder_layout)

        # Add stretch
        layout.addStretch()

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setMaximumWidth(100)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        button_layout.addStretch()

        finish_btn = QPushButton("Finish")
        finish_btn.setMaximumWidth(100)
        finish_btn.clicked.connect(self.on_finish_clicked)
        button_layout.addWidget(finish_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def browse_folder(self):
        """Open folder browser dialog for project folder."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Project Folder", str(Path.home())
        )
        if folder:
            self.folder_input.setText(folder)

    def browse_images(self):
        """Open file/folder browser for training images."""
        # First try to browse for a directory
        folder = QFileDialog.getExistingDirectory(
            self, "Select Training Images Folder", str(Path.home())
        )
        if folder:
            self.images_input.setText(folder)
        else:
            # If user cancels, offer to select a file
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Training Image File",
                str(Path.home()),
                "Image Files (*.jpg *.jpeg *.png *.bmp);;All Files (*)",
            )
            if file_path:
                self.images_input.setText(file_path)

    def on_finish_clicked(self):
        """Validate and create project."""
        # Validate required fields
        project_name = self.name_input.text().strip()
        project_folder = self.folder_input.text().strip()

        if not project_name:
            self._show_error("Project name is required")
            return

        if not project_folder:
            self._show_error("Project folder is required")
            return

        # Check if folder exists or can be created
        folder_path = Path(project_folder)
        try:
            folder_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self._show_error(f"Cannot create folder: {e}")
            return

        # Prepare project data
        self.project_data = {
            "name": project_name,
            "description": self.description_input.toPlainText().strip(),
            "type": self.type_combo.currentText(),
            "folder": str(folder_path.absolute()),
        }

        self.project_created.emit(self.project_data)
        self.accept()

    def _show_error(self, message: str):
        """Show error message (simple implementation)."""
        from PyQt6.QtWidgets import QMessageBox

        QMessageBox.warning(self, "Validation Error", message)

    def get_project_data(self) -> Optional[Dict]:
        """Get the created project data."""
        return self.project_data

    def set_theme(self, theme: str):
        """
        Switch theme at runtime.

        Args:
            theme: "light" or "dark"
        """
        self.current_theme = theme
        apply_theme(self, theme)
