"""
Main Window (PyQt6) - Multi-tab interface for CALQA project workflow
Tabs: segmentation, batch processing, results, benchmark
"""

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTabWidget,
    QPushButton,
    QProgressBar,
    QRadioButton,
    QButtonGroup,
    QLineEdit,
    QComboBox,
    QTableWidget,
    QTableWidgetItem,
    QFileDialog,
    QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
from pathlib import Path
from typing import Dict

from view.themes import apply_theme


class MainWindowPyQt(QMainWindow):
    """
    Main application window with multi-tab interface for project workflows.
    """

    # Signals for controller communication
    compute_clicked = pyqtSignal(str)          # method: "CPU_LoG", "CPU_DoG", etc.
    batch_process_clicked = pyqtSignal(str, str)   # folder_path, method
    export_results_clicked = pyqtSignal(str)       # file path
    benchmark_compare_clicked = pyqtSignal(str, str)  # method1, method2
    validate_ssim_clicked = pyqtSignal()           # on-demand SSIM validation

    def __init__(self, project_data: Dict, theme: str = "light"):
        super().__init__()
        self.current_theme = theme
        self.project_data = project_data
        self.current_image = None
        self.current_result_image = None
        self.processing_time_ms = 0.0

        self.init_ui()
        apply_theme(self, theme)

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle(f"CALQA – {self.project_data.get('name', 'Project')}")
        self.setGeometry(100, 100, 1000, 700)

        # Create menu bar
        self._create_menu_bar()

        # Create central widget with tabs
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()

        # Tab widget
        self.tabs = QTabWidget()

        # Add tabs
        self.segmentation_tab = self._create_segmentation_tab()
        self.batch_tab = self._create_batch_tab()
        self.results_tab = self._create_results_tab()
        self.benchmark_tab = self._create_benchmark_tab()

        self.tabs.addTab(self.segmentation_tab, "Segmentation")
        self.tabs.addTab(self.batch_tab, "Batch Processing")
        self.tabs.addTab(self.results_tab, "Results")
        self.tabs.addTab(self.benchmark_tab, "Benchmark")

        main_layout.addWidget(self.tabs)

        central_widget.setLayout(main_layout)

        # Create status bar
        self.statusBar().showMessage("Ready")

    def _create_menu_bar(self):
        """Create the menu bar with standard actions."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

        # Help menu
        help_menu = menubar.addMenu("Help")
        about_action = help_menu.addAction("About CALQA")
        about_action.triggered.connect(self._show_about)

    def _create_segmentation_tab(self) -> QWidget:
        """Create Segmentation tab."""
        widget = QWidget()
        layout = QVBoxLayout()

        # Image preview
        layout.addWidget(QLabel("Image Preview"))
        self.image_preview_label = QLabel("No image loaded")
        self.image_preview_label.setMinimumHeight(250)
        self.image_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview_label.setStyleSheet(
            "border: 1px solid #8B5CF6; background-color: #F9FAFB;"
        )
        layout.addWidget(self.image_preview_label)

        # Load image button
        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self.load_image)
        layout.addWidget(load_btn)

        # Device selection
        layout.addWidget(QLabel("Select Device"))
        device_layout = QHBoxLayout()

        self.device_group = QButtonGroup()
        cpu_radio = QRadioButton("CPU")
        cpu_radio.setChecked(True)
        gpu_radio = QRadioButton("GPU")

        self.device_group.addButton(cpu_radio, 0)
        self.device_group.addButton(gpu_radio, 1)

        device_layout.addWidget(cpu_radio)
        device_layout.addWidget(gpu_radio)
        layout.addLayout(device_layout)

        # Filter selection
        layout.addWidget(QLabel("Select Filter"))
        filter_layout = QHBoxLayout()

        self.filter_group = QButtonGroup()
        log_radio = QRadioButton("Laplacian of Gaussian (LoG)")
        log_radio.setChecked(True)
        dog_radio = QRadioButton("Difference of Gaussian (DoG)")

        self.filter_group.addButton(log_radio, 0)
        self.filter_group.addButton(dog_radio, 1)

        filter_layout.addWidget(log_radio)
        filter_layout.addWidget(dog_radio)
        layout.addLayout(filter_layout)

        # Progress bar (hidden initially)
        self.segmentation_progress = QProgressBar()
        self.segmentation_progress.setVisible(False)
        layout.addWidget(self.segmentation_progress)

        # Compute button
        compute_btn = QPushButton("Compute")
        compute_btn.clicked.connect(self._on_compute_clicked)
        layout.addWidget(compute_btn)

        # Result display
        layout.addWidget(QLabel("Result"))
        self.result_image_label = QLabel("Result will appear here")
        self.result_image_label.setMinimumHeight(250)
        self.result_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_image_label.setStyleSheet(
            "border: 1px solid #8B5CF6; background-color: #F9FAFB;"
        )
        layout.addWidget(self.result_image_label)

        # Time display
        time_layout = QHBoxLayout()
        self.time_label = QLabel("Processing Time: —")
        time_layout.addWidget(self.time_label)
        time_layout.addStretch()
        layout.addLayout(time_layout)

        widget.setLayout(layout)
        return widget

    def _create_batch_tab(self) -> QWidget:
        """Create Batch Processing tab."""
        widget = QWidget()
        layout = QVBoxLayout()

        # Folder selection
        layout.addWidget(QLabel("Images Folder"))
        folder_layout = QHBoxLayout()

        self.batch_folder_input = QLineEdit()
        self.batch_folder_input.setReadOnly(True)
        self.batch_folder_input.setPlaceholderText(
            "Select folder containing images to process"
        )

        browse_btn = QPushButton("Browse...")
        browse_btn.setMaximumWidth(100)
        browse_btn.clicked.connect(self._browse_batch_folder)

        folder_layout.addWidget(self.batch_folder_input)
        folder_layout.addWidget(browse_btn)
        layout.addLayout(folder_layout)

        # Device + filter selection
        layout.addWidget(QLabel("Select Method"))
        method_layout = QHBoxLayout()

        self.batch_device_group = QButtonGroup()
        batch_cpu_radio = QRadioButton("CPU")
        batch_cpu_radio.setChecked(True)
        batch_gpu_radio = QRadioButton("GPU")
        self.batch_device_group.addButton(batch_cpu_radio, 0)
        self.batch_device_group.addButton(batch_gpu_radio, 1)

        self.batch_filter_group = QButtonGroup()
        batch_log_radio = QRadioButton("LoG")
        batch_log_radio.setChecked(True)
        batch_dog_radio = QRadioButton("DoG")
        self.batch_filter_group.addButton(batch_log_radio, 0)
        self.batch_filter_group.addButton(batch_dog_radio, 1)

        method_layout.addWidget(batch_cpu_radio)
        method_layout.addWidget(batch_gpu_radio)
        method_layout.addWidget(batch_log_radio)
        method_layout.addWidget(batch_dog_radio)
        layout.addLayout(method_layout)

        # Process button
        process_btn = QPushButton("Process All")
        process_btn.clicked.connect(self._on_batch_process_clicked)
        layout.addWidget(process_btn)

        # Progress bar
        self.batch_progress = QProgressBar()
        self.batch_progress.setVisible(False)
        layout.addWidget(self.batch_progress)

        # Results table
        layout.addWidget(QLabel("Batch Results"))
        self.batch_table = QTableWidget()
        self.batch_table.setColumnCount(4)
        self.batch_table.setHorizontalHeaderLabels(
            ["Filename", "Method", "Time (ms)", "Status"]
        )
        self.batch_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.batch_table)

        widget.setLayout(layout)
        return widget

    def _create_results_tab(self) -> QWidget:
        """Create Results tab."""
        widget = QWidget()
        layout = QVBoxLayout()

        # Results table — pivot: one row per filename, one col per method
        layout.addWidget(QLabel("Saved Results"))
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(7)
        self.results_table.setHorizontalHeaderLabels(
            ["Filename", "CPU_LoG (ms)", "CPU_DoG (ms)", "GPU_LoG (ms)", "GPU_DoG (ms)", "SSIM", "Density"]
        )
        self.results_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.results_table)

        # Action buttons row
        action_layout = QHBoxLayout()

        ssim_btn = QPushButton("Validate Accuracy (SSIM)")
        ssim_btn.setToolTip(
            "Re-process each result with CPU DoG + GPU DoG and compute SSIM score"
        )
        ssim_btn.clicked.connect(lambda: self.validate_ssim_clicked.emit())
        action_layout.addWidget(ssim_btn)

        action_layout.addStretch()

        csv_btn = QPushButton("Export as CSV")
        csv_btn.clicked.connect(lambda: self._on_export_clicked("csv"))
        action_layout.addWidget(csv_btn)

        json_btn = QPushButton("Export as JSON")
        json_btn.clicked.connect(lambda: self._on_export_clicked("json"))
        action_layout.addWidget(json_btn)

        delete_btn = QPushButton("Delete Selected")
        delete_btn.clicked.connect(self._on_delete_result_clicked)
        action_layout.addWidget(delete_btn)

        layout.addLayout(action_layout)

        widget.setLayout(layout)
        return widget

    def _create_benchmark_tab(self) -> QWidget:
        """Create Benchmark tab for comparing methods."""
        widget = QWidget()
        layout = QVBoxLayout()

        # Method selection
        layout.addWidget(QLabel("Compare Methods"))
        method_layout = QHBoxLayout()

        # Method 1 dropdown
        method1_label = QLabel("Method Filter 1:")
        method_layout.addWidget(method1_label)
        self.benchmark_method1_combo = QComboBox()
        self.benchmark_method1_combo.setEditable(False)
        self.benchmark_method1_combo.addItems(["CPU_LoG", "CPU_DoG", "GPU_LoG", "GPU_DoG"])
        method_layout.addWidget(self.benchmark_method1_combo)

        # Method 2 dropdown
        method2_label = QLabel("Method Filter 2:")
        method_layout.addWidget(method2_label)
        self.benchmark_method2_combo = QComboBox()
        self.benchmark_method2_combo.setEditable(False)
        self.benchmark_method2_combo.addItems(["CPU_LoG", "CPU_DoG", "GPU_LoG", "GPU_DoG"])
        method_layout.addWidget(self.benchmark_method2_combo)

        apply_btn = QPushButton("Apply Comparison")
        apply_btn.clicked.connect(self._on_benchmark_apply_clicked)
        method_layout.addWidget(apply_btn)

        layout.addLayout(method_layout)

        # Comparison table
        layout.addWidget(QLabel("Benchmark Comparison"))
        self.benchmark_table = QTableWidget()
        self.benchmark_table.setColumnCount(4)
        self.benchmark_table.setHorizontalHeaderLabels(
            ["Filename", "Method 1 (ms)", "Method 2 (ms)", "Speedup"]
        )
        self.benchmark_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.benchmark_table)

        # Summary
        summary_layout = QHBoxLayout()
        self.benchmark_summary_label = QLabel("")
        summary_layout.addWidget(self.benchmark_summary_label)
        summary_layout.addStretch()
        layout.addLayout(summary_layout)

        widget.setLayout(layout)
        return widget

    def _on_benchmark_apply_clicked(self):
        """Emit benchmark compare signal."""
        method1 = self.benchmark_method1_combo.currentText().strip()
        method2 = self.benchmark_method2_combo.currentText().strip()
        if not method1 or not method2:
            QMessageBox.warning(self, "Error", "Please enter both methods to compare")
            return
        self.benchmark_compare_clicked.emit(method1, method2)

    def load_image(self):
        """Open file dialog to load an image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Image",
            str(Path.home()),
            "Image Files (*.jpg *.jpeg *.png *.bmp);;All Files (*)",
        )
        if file_path:
            self.image_preview_label.setText(f"Loaded: {Path(file_path).name}")
            # Store for later use
            self.current_image_path = file_path
            # Show preview (simple implementation)
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                scaled = pixmap.scaledToHeight(
                    200, Qt.TransformationMode.SmoothTransformation
                )
                self.image_preview_label.setPixmap(scaled)

    def set_result_image(self, image_array):
        """Display result image from numpy array."""
        try:
            import numpy as np

            if image_array is None:
                return

            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)

            # Ensure contiguous memory for Qt
            image_array = np.ascontiguousarray(image_array)

            if image_array.ndim == 2:
                height, width = image_array.shape
                qimage = QImage(
                    image_array.data,
                    width,
                    height,
                    width,
                    QImage.Format.Format_Grayscale8,
                )
            elif image_array.ndim == 3 and image_array.shape[2] == 3:
                height, width, _ = image_array.shape
                qimage = QImage(
                    image_array.data,
                    width,
                    height,
                    3 * width,
                    QImage.Format.Format_RGB888,
                )
            elif image_array.ndim == 3 and image_array.shape[2] == 4:
                height, width, _ = image_array.shape
                qimage = QImage(
                    image_array.data,
                    width,
                    height,
                    4 * width,
                    QImage.Format.Format_RGBA8888,
                )
            else:
                return

            pixmap = QPixmap.fromImage(qimage)
            if not pixmap.isNull():
                scaled = pixmap.scaledToHeight(
                    200, Qt.TransformationMode.SmoothTransformation
                )
                self.result_image_label.setPixmap(scaled)
                self.current_result_image = image_array
        except Exception:
            # Fail silently if the image cannot be displayed
            pass

    def set_processing_time(self, time_ms: float):
        """Update processing time display."""
        self.processing_time_ms = time_ms
        self.time_label.setText(f"Processing Time: {time_ms:.2f} ms")

    def show_progress(self, visible: bool = True):
        """Show/hide progress bar on segmentation tab."""
        self.segmentation_progress.setVisible(visible)

    def update_batch_table(self, rows):
        """Update batch results table."""
        self.batch_table.setRowCount(len(rows))
        for row_idx, row_data in enumerate(rows):
            for col_idx, value in enumerate(row_data):
                item = QTableWidgetItem(str(value))
                self.batch_table.setItem(row_idx, col_idx, item)

    def update_results_table(self, rows):
        """Update results table."""
        self.results_table.setRowCount(len(rows))
        for row_idx, row_data in enumerate(rows):
            for col_idx, value in enumerate(row_data):
                item = QTableWidgetItem(str(value))
                self.results_table.setItem(row_idx, col_idx, item)

    def set_benchmark_methods(self, methods):
        """Populate benchmark method selection options."""
        self.benchmark_method1_combo.clear()
        self.benchmark_method2_combo.clear()
        self.benchmark_method1_combo.addItems(methods)
        self.benchmark_method2_combo.addItems(methods)

    def update_benchmark_table(self, rows, summary_text: str = ""):
        """Update benchmark comparison table."""
        self.benchmark_table.setRowCount(len(rows))
        for row_idx, row_data in enumerate(rows):
            for col_idx, value in enumerate(row_data):
                item = QTableWidgetItem(str(value))
                self.benchmark_table.setItem(row_idx, col_idx, item)
        self.benchmark_summary_label.setText(summary_text)

    def _on_compute_clicked(self):
        """Handle compute button click."""
        if not hasattr(self, "current_image_path"):
            QMessageBox.warning(self, "Error", "Please load an image first")
            return

        # Determine device (CPU/GPU) and filter
        device = "CPU" if self.device_group.checkedId() == 0 else "GPU"
        filter_name = "LoG" if self.filter_group.checkedId() == 0 else "DoG"
        method = f"{device}_{filter_name}"

        self.show_progress(True)
        self.statusBar().showMessage(f"Processing with {method}...")
        self.compute_clicked.emit(method)

    def _on_batch_process_clicked(self):
        """Handle batch process button click."""
        folder = self.batch_folder_input.text().strip()
        if not folder:
            QMessageBox.warning(self, "Error", "Please select a folder first")
            return

        # Determine method (device + filter)
        device = "CPU" if self.batch_device_group.checkedId() == 0 else "GPU"
        filter_name = "LoG" if self.batch_filter_group.checkedId() == 0 else "DoG"
        method = f"{device}_{filter_name}"

        self.batch_progress.setVisible(True)
        self.statusBar().showMessage("Processing batch...")
        self.batch_process_clicked.emit(folder, method)

    def _on_export_clicked(self, format_type: str):
        """Handle export button click."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            str(Path.home()),
            f"{'CSV' if format_type == 'csv' else 'JSON'} Files (*.{'csv' if format_type == 'csv' else 'json'})",
        )
        if file_path:
            self.export_results_clicked.emit(file_path)
            self.statusBar().showMessage(f"Results exported to {file_path}")

    def _on_delete_result_clicked(self):
        """Handle delete result button click."""
        current_row = self.results_table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "Error", "Please select a result to delete")
            return

        reply = QMessageBox.question(
            self, "Confirm", "Delete selected result?", QMessageBox.StandardButton.Yes
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.results_table.removeRow(current_row)

    def _browse_batch_folder(self):
        """Open folder browser for batch processing."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Images Folder", str(Path.home())
        )
        if folder:
            self.batch_folder_input.setText(folder)

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.information(
            self,
            "About CALQA",
            "CALQA – Centella Asiatica Leaves Quality Assessment\n"
            "GPU-Accelerated Vein Segmentation\n\n"
            "Version: 1.0\n"
            "University of Westminster",
        )

    def set_status_message(self, message: str):
        """Update status bar message."""
        self.statusBar().showMessage(message)
