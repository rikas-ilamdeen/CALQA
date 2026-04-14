from model.image_loader import load_image
from model.preprocessing import to_grayscale
from model.cpu.log_cpu import apply_log as apply_log_cpu
from model.cpu.dog_cpu import apply_dog as apply_dog_cpu
from model.gpu.log_gpu import apply_log as apply_log_gpu
from model.gpu.dog_gpu import apply_dog as apply_dog_gpu
from model.timer import Timer
from model.project_manager import ProjectManager
from typing import Optional


class AppController:
    """
    Controller layer coordinating View <-> Model data flow.

    Primarily used by the PyQt6 interface, with compatibility hooks
    for views exposing `show_result` and `show_time` methods.
    Main responsibilities:
    - load input images
    - run selected CPU/GPU algorithm
    - keep last processing time
    - provide batch processing utilities
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

        # Compatibility hook for views that can preview loaded images.
        if hasattr(self.view, "show_result"):
            self.view.show_result(self.image)

    def process(self, method):
        """Process image with selected method.

        Supports CPU and GPU methods with naming like:
        - "CPU_LoG", "CPU_DoG" (CPU processing)
        - "GPU_LoG", "GPU_DoG" (GPU processing)
        - Legacy: "LoG", "DoG" (defaults to CPU)

        Args:
            method: Method name including device prefix.
        """
        # 1) Parse method string into device + algorithm.
        device = "CPU"  # default
        alg = None

        if method is None:
            alg = None
        elif method.upper().startswith("GPU_"):
            device = "GPU"
            filter_part = method[4:]  # Remove "GPU_" prefix
            if filter_part.upper() == "LOG":
                alg = "LoG"
            elif filter_part.upper() == "DOG":
                alg = "DoG"
        elif method.upper().startswith("CPU_"):
            device = "CPU"
            filter_part = method[4:]  # Remove "CPU_" prefix
            if filter_part.upper() == "LOG":
                alg = "LoG"
            elif filter_part.upper() == "DOG":
                alg = "DoG"
        else:
            # Legacy support
            device = "CPU"
            if method.upper() == "LOG":
                alg = "LoG"
            elif method.upper() == "DOG":
                alg = "DoG"

        # 2) Run the processing pipeline and measure total elapsed time.
        with Timer() as t:
            # Reload image
            image = load_image(self.image_path)

            # Convert to grayscale (required for edge detection)
            gray = to_grayscale(image)

            # 3) Execute selected implementation (CPU or GPU).
            gpu_kernel_time = None
            if device == "GPU":
                if alg == "LoG":
                    output, gpu_kernel_time = apply_log_gpu(gray)
                elif alg == "DoG":
                    output, gpu_kernel_time = apply_dog_gpu(gray)
                else:
                    output = gray
            else:  # CPU
                if alg == "LoG":
                    output = apply_log_cpu(gray)
                elif alg == "DoG":
                    output = apply_dog_cpu(gray)
                else:
                    output = gray

            # Store result
            self.image = output

            # Compatibility hook for views that display processed output.
            if hasattr(self.view, "show_result"):
                self.view.show_result(output)

        # 4) Store timing: GPU returns kernel time; CPU uses full elapsed time.
        if gpu_kernel_time is not None:
            self._last_processing_time = gpu_kernel_time
        else:
            self._last_processing_time = t.elapsed_ms

        # Compatibility hook for views that display processing time.
        if hasattr(self.view, "show_time"):
            self.view.show_time(t.elapsed_ms)

    def batch_process_folder(self, folder_path, method="CPU_LoG"):
        """
        Process all images in a folder.

        Args:
            folder_path: Path to folder containing images
            method: Method name like "CPU_LoG", "GPU_DoG", etc.

        Returns:
            List of results with metadata
        """
        from pathlib import Path

        results = []
        folder = Path(folder_path)

        # Collect supported image files from folder.
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
