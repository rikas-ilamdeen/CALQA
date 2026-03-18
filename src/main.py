"""
CALQA Main Entry Point - PyQt6 Application Launcher
Centella Asiatica Leaves Quality Assessment
GPU-Accelerated Vein Segmentation and Quality Analysis
"""

import sys
from PyQt6.QtWidgets import QApplication

from controller.project_controller import ProjectController


def main():
    """
    Initialize and run CALQA application.
    Uses PyQt6 with welcome screen → project creation/loading → main window workflow.
    """
    # Create QApplication instance
    app = QApplication(sys.argv)

    # Set application metadata
    app.setApplicationName("CALQA")
    app.setApplicationVersion("1.0")

    # Create project controller
    project_controller = ProjectController(theme="light")

    # Show welcome screen
    project_controller.show_welcome()

    # Run application event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
