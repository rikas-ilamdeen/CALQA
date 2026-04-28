"""
Project Controller - High-level app workflow controller for CALQA
Manages project creation, loading, and main window lifecycle
"""

import logging
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox
from pathlib import Path

from model.project_manager import ProjectManager
from model import feature_extractor
from model.ssim_validator import compute_ssim
from view.welcome_screen import WelcomeScreen
from view.new_project_dialog import NewProjectDialog
from view.main_window_pyqt import MainWindowPyQt
from controller.app_controller import AppController

logger = logging.getLogger(__name__)

class ProjectController:
    """
    High-level controller managing project workflows.
    Orchestrates project lifecycle and UI transitions:
    welcome -> create/open project -> main window operations.
    """

    def __init__(self, theme: str = "light"):
        self.theme = theme
        self.project_manager = ProjectManager()
        self.app_controller = None
        self.current_main_window = None
        self.welcome_screen = None

    def show_welcome(self):
        """Display the welcome screen."""
        self.welcome_screen = WelcomeScreen(theme=self.theme)
        self.welcome_screen.create_project_clicked.connect(self.on_create_project_clicked)
        self.welcome_screen.open_project_clicked.connect(self.on_open_project_clicked)
        self.welcome_screen.show()

    def on_create_project_clicked(self):
        """Handle create project button click."""
        dialog = NewProjectDialog(parent=self.welcome_screen, theme=self.theme)
        dialog.project_created.connect(self.on_project_created)

        if dialog.exec() == NewProjectDialog.DialogCode.Accepted:
            project_data = dialog.get_project_data()
            if project_data:
                # Create project using project manager
                success = self.project_manager.create_project(
                    name=project_data["name"],
                    description=project_data["description"],
                    project_type=project_data["type"],
                    parent_folder=project_data["folder"],
                )

                if success:
                    # Hide welcome screen
                    if self.welcome_screen:
                        self.welcome_screen.hide()

                    # Open main window
                    self._open_main_window()
                else:
                    QMessageBox.critical(self.welcome_screen, "Error", "Failed to create project")

    def on_open_project_clicked(self):
        """Handle open project button click."""
        file_path, _ = QFileDialog.getOpenFileName(
            self.welcome_screen,
            "Open Project",
            str(Path.home()),
            "Project Files (project.json);;All Files (*)",
        )

        if file_path:
            if self.project_manager.load_project(file_path):
                # Hide welcome screen
                if self.welcome_screen:
                    self.welcome_screen.hide()

                # Open main window
                self._open_main_window()
            else:
                QMessageBox.critical(
                    self.welcome_screen, "Error", f"Failed to load project: {file_path}"
                )

    def on_project_created(self, project_data):
        """Handle project created signal from dialog."""
        # Already handled in on_create_project_clicked
        pass

    def _open_main_window(self):
        """Open the main application window."""
        if not self.project_manager.current_project:
            QMessageBox.critical(self.welcome_screen, "Error", "No project loaded")
            return

        project_info = self.project_manager.get_project_info()

        # Create main window
        self.current_main_window = MainWindowPyQt(project_info, theme=self.theme)

        # Create app controller
        self.app_controller = AppController(self.current_main_window)
        self.app_controller.project_manager = self.project_manager

        # Connect UI signals to controller handlers.
        self.current_main_window.compute_clicked.connect(
            self.on_compute_clicked
        )
        self.current_main_window.batch_process_clicked.connect(
            self.on_batch_process_clicked
        )
        self.current_main_window.export_results_clicked.connect(
            self.on_export_results_clicked
        )
        self.current_main_window.benchmark_compare_clicked.connect(
            self.on_benchmark_compare_clicked
        )
        self.current_main_window.validate_ssim_clicked.connect(
            self.on_validate_ssim_clicked
        )

        # Load and display results
        self._update_results_display()

        # Show main window
        self.current_main_window.show()

    def on_compute_clicked(self, method: str):
        """Handle compute button click from main window."""
        if not hasattr(self.current_main_window, "current_image_path"):
            QMessageBox.warning(self.current_main_window, "Error", "Please load an image first")
            self.current_main_window.show_progress(False)
            return

        try:
            image_path = self.current_main_window.current_image_path

            # Use AppController to process (method includes device prefix like "GPU_LoG")
            self.app_controller.load_image(image_path)
            self.app_controller.process(method)

            # Display result
            if self.app_controller.image is not None:
                self.current_main_window.set_result_image(self.app_controller.image)

            # Display processing time in Segmentation tab only (not saved to Results)
            self.current_main_window.set_processing_time(
                self.app_controller._last_processing_time
            )
            self.current_main_window.show_progress(False)
            self.current_main_window.set_status_message("Segmentation complete")

        except Exception as e:
            QMessageBox.critical(self.current_main_window, "Error", f"Processing failed: {e}")
            self.current_main_window.show_progress(False)

    def on_batch_process_clicked(self, folder_path: str, method: str):
        """Handle batch processing."""
        try:
            from pathlib import Path

            folder = Path(folder_path)
            image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))

            if not image_files:
                QMessageBox.warning(
                    self.current_main_window, "Error", "No image files found in folder"
                )
                self.current_main_window.batch_progress.setVisible(False)
                return

            # Process each image
            results = []
            total_time = 0.0

            for idx, image_file in enumerate(image_files):
                # Update progress
                progress = int((idx / len(image_files)) * 100)
                self.current_main_window.batch_progress.setValue(progress)
                QApplication.processEvents()

                try:
                    # Process image with selected method
                    self.app_controller.load_image(str(image_file))
                    self.app_controller.process(method)
                    output = self.app_controller.image

                    # Save output image
                    output_path = self.project_manager.save_output_image(
                        method, str(image_file), output
                    )

                    # --- Feature extraction ---
                    features = feature_extractor.extract(output)

                    # Record display row
                    results.append(
                        [
                            image_file.name,
                            method,
                            f"{self.app_controller._last_processing_time:.2f}",
                            "Success",
                        ]
                    )
                    total_time += self.app_controller._last_processing_time

                    # Save result metadata (including features)
                    self.project_manager.save_result(
                        {
                            "method": method,
                            "input_file": str(image_file),
                            "output_file": output_path or "",
                            "processing_time_ms": self.app_controller._last_processing_time,
                            "status": "success",
                            "vein_density": features["vein_density"],
                            "edge_count": features["edge_count"],
                            "ssim_score": None,
                        }
                    )

                except Exception as e:
                    results.append([image_file.name, method, "—", f"Error: {str(e)[:20]}"])

            # Update batch table
            self.current_main_window.update_batch_table(results)
            self.current_main_window.batch_progress.setValue(100)
            self.current_main_window.set_status_message(
                f"Batch processing complete. Total time: {total_time:.2f} ms"
            )

            # Refresh Results tab
            self._update_results_display()

        except Exception as e:
            QMessageBox.critical(self.current_main_window, "Error", f"Batch processing failed: {e}")
        finally:
            self.current_main_window.batch_progress.setVisible(False)

    def on_export_results_clicked(self, file_path: str):
        """Handle export results signal."""
        try:
            if file_path.endswith(".csv"):
                success = self.project_manager.export_results_csv(file_path)
                format_name = "CSV"
            elif file_path.endswith(".json"):
                success = self.project_manager.export_results_json(file_path)
                format_name = "JSON"
            else:
                QMessageBox.warning(
                    self.current_main_window, "Error", "Unsupported file format"
                )
                return

            if success:
                QMessageBox.information(
                    self.current_main_window,
                    "Success",
                    f"Results exported as {format_name}",
                )
            else:
                QMessageBox.warning(self.current_main_window, "Error", "No results to export")

        except Exception as e:
            QMessageBox.critical(self.current_main_window, "Error", f"Export failed: {e}")

    def _update_results_display(self):
        """Refresh Results tab using a pivot layout.

        Each row corresponds to one input file, with fixed columns for
        CPU/GPU LoG/DoG timings plus SSIM and vein-density summaries.
        """
        results = self.project_manager.get_results()

        if not results:
            self.current_main_window.update_results_table([])
            self.current_main_window.set_benchmark_methods([])
            return

        # Build pivot map: pivot[filename][method] = result_dict.
        METHODS = ["CPU_LoG", "CPU_DoG", "GPU_LoG", "GPU_DoG"]
        pivot = {}
        for r in results:
            fname = Path(r.get("input_file", "")).name
            method = r.get("method", "")
            if fname not in pivot:
                pivot[fname] = {}
            pivot[fname][method] = r

        table_data = []
        for fname in sorted(pivot.keys()):
            row_data = pivot[fname]

            # Collect times in the fixed column order used by the table.
            method_cells = []
            for m in METHODS:
                if m in row_data:
                    t = row_data[m].get("processing_time_ms", 0)
                    method_cells.append(f"{t:.2f}")
                else:
                    method_cells.append("—")

            # SSIM is meaningful for DoG CPU/GPU comparison.
            ssim_val = None
            for m in ["CPU_DoG", "GPU_DoG"]:
                if m in row_data and row_data[m].get("ssim_score") is not None:
                    ssim_val = row_data[m]["ssim_score"]
                    break
            ssim_str = f"{ssim_val:.4f}" if ssim_val is not None else "N/A"

            # Prefer density from DoG first, then LoG as fallback.
            density_val = None
            for m in ["GPU_DoG", "CPU_DoG", "GPU_LoG", "CPU_LoG"]:
                if m in row_data and row_data[m].get("vein_density") is not None:
                    density_val = row_data[m]["vein_density"]
                    break
            density_str = f"{density_val:.2f}" if density_val is not None else "N/A"

            table_data.append([fname] + method_cells + [ssim_str, density_str])

        self.current_main_window.update_results_table(table_data)

        # Update available methods for benchmark comparison
        methods = sorted({r.get("method") for r in results if r.get("method")})
        self.current_main_window.set_benchmark_methods(methods)


    def on_validate_ssim_clicked(self):
        """On-demand SSIM validation: re-processes each stored result with CPU_DoG
        and GPU_DoG, computes SSIM, and updates project.json.
        Only DoG results get a real SSIM score; LoG results stay N/A.
        """
        results = self.project_manager.get_results()
        if not results:
            QMessageBox.information(
                self.current_main_window, "SSIM", "No results to validate."
            )
            return

        self.current_main_window.set_status_message("Computing SSIM scores...")
        QApplication.processEvents()

        updated = 0
        for r in results:
            method = r.get("method", "")
            input_file = r.get("input_file", "")

            # Only compute SSIM for DoG methods
            if "DoG" not in method:
                continue

            try:
                # Run CPU_DoG
                self.app_controller.load_image(input_file)
                self.app_controller.process("CPU_DoG")
                cpu_out = self.app_controller.image

                # Run GPU_DoG
                self.app_controller.load_image(input_file)
                self.app_controller.process("GPU_DoG")
                gpu_out = self.app_controller.image

                score = compute_ssim(cpu_out, gpu_out)
                r["ssim_score"] = score
                self.project_manager.save_result(r)
                updated += 1

            except Exception as e:
                logger.warning("[SSIM] Skipping %s: %s", input_file, e)

        self._update_results_display()
        self.current_main_window.set_status_message(
            f"SSIM validation complete. Updated {updated} result(s)."
        )

    def on_benchmark_compare_clicked(self, method1: str, method2: str):
        """Compare two methods using stored results."""
        results = self.project_manager.get_results()

        # Build lookup by (filename, method)
        lookup = {}
        for r in results:
            key = (Path(r.get("input_file", "")).name, r.get("method", ""))
            lookup[key] = float(r.get("processing_time_ms", 0))

        rows = []
        total1 = 0.0
        total2 = 0.0
        speedups = []

        filenames = sorted({k[0] for k in lookup.keys()})
        for fname in filenames:
            t1 = lookup.get((fname, method1))
            t2 = lookup.get((fname, method2))
            if t1 is None or t2 is None:
                continue

            speedup = t1 / t2 if t2 > 0 else None
            rows.append([
                fname,
                f"{t1:.2f}",
                f"{t2:.2f}",
                f"{speedup:.2f}" if speedup is not None else "—",
            ])
            total1 += t1
            total2 += t2
            if speedup is not None:
                speedups.append(speedup)

        avg_speedup = sum(speedups) / len(speedups) if speedups else 0.0
        summary = (
            f"Total {method1}: {total1:.2f} ms  |  "
            f"Total {method2}: {total2:.2f} ms  |  "
            f"Avg Speedup: {avg_speedup:.2f}"
        )

        self.current_main_window.update_benchmark_table(rows, summary)

    def set_theme(self, theme: str):
        """Change theme globally."""
        self.theme = theme

        if self.welcome_screen:
            self.welcome_screen.set_theme(theme)

        if self.current_main_window:
            self.current_main_window.set_theme(theme)
