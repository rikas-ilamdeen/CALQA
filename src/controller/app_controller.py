from model.image_loader import load_image
from model.preprocessing import to_grayscale
from model.cpu.log_cpu import apply_log
from model.cpu.dog_cpu import apply_dog
from model.timer import Timer
from model.project_manager import ProjectManager
from typing import Optional


class AppController:
    """
    Controller layer: coordinates data flow between View and Model
    Supports both Tkinter (legacy) and PyQt6 (new) interfaces.
    Manages image processing, timing, and project workflows.
    """

    def __init__(self, view):
        self.view = view
        self.image_path = None
        self.image = None
        self.project_manager: Optional[ProjectManager] = None
        self._last_processing_time = 0.0

    def load_image(self, path):
        """
        Load image from file path.

        Args:
            path: Path to image file
        """
        self.image_path = path
        self.image = load_image(path)

        # For PyQt6 views with show_result method
        if hasattr(self.view, "show_result"):
            self.view.show_result(self.image)

    def process(self, method):
        """Process image with selected method.

        Supports both the legacy method names ("LoG", "DoG") and the
        new project method naming ("CPU_LoG", "CPU_DoG").

        Args:
            method: "CPU_LoG", "CPU_DoG", "LoG", or "DoG".
        """
        # Normalize method names to the core algorithm
        alg = None
        if method is None:
            alg = None
        elif method.upper().endswith("LOG"):
            alg = "LoG"
        elif method.upper().endswith("DOG"):
            alg = "DoG"

        # Measure time
        with Timer() as t:
            # Reload image
            image = load_image(self.image_path)

            # Convert to grayscale (required for edge detection)
            gray = to_grayscale(image)

            # Select CPU-based filter
            if alg == "LoG":
                output = apply_log(gray)
            elif alg == "DoG":
                output = apply_dog(gray)
            else:
                output = gray

            # Store result
            self.image = output

            # Display result (for Tkinter views)
            if hasattr(self.view, "show_result"):
                self.view.show_result(output)

        # Store processing time
        self._last_processing_time = t.elapsed_ms

        # Show time (for Tkinter views)
        if hasattr(self.view, "show_time"):
            self.view.show_time(t.elapsed_ms)

    def batch_process_folder(self, folder_path, method="LoG"):
        """
        Process all images in a folder.

        Args:
            folder_path: Path to folder containing images
            method: "LoG" or "DoG"

        Returns:
            List of results with metadata
        """
        from pathlib import Path

        results = []
        folder = Path(folder_path)

        # Find all image files
        image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))

        for image_file in image_files:
            try:
                self.load_image(str(image_file))
                self.process(method)

                results.append(
                    {
                        "filename": image_file.name,
                        "method": method,
                        "processing_time_ms": self._last_processing_time,
                        "status": "success",
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "filename": image_file.name,
                        "method": method,
                        "processing_time_ms": 0.0,
                        "status": f"error: {str(e)}",
                    }
                )

        return results

    def save_current_result(self, method: str, input_file: str) -> bool:
        """Save current result to project.

        Args:
            method: "CPU_LoG" or "CPU_DoG" (or legacy "LoG"/"DoG")
            input_file: Path to input image file

        Returns:
            True if saved successfully
        """
        if not self.project_manager:
            return False

        # Derive a friendly output filename based on input and method
        from pathlib import Path

        base_name = Path(input_file).stem
        safe_method = method.replace(" ", "_")
        output_filename = f"{base_name}_{safe_method}.png"

        result_data = {
            "timestamp": None,
            "method": method,
            "input_file": input_file,
            "output_file": output_filename,
            "processing_time_ms": self._last_processing_time,
            "status": "success",
        }

        return self.project_manager.save_result(result_data)
