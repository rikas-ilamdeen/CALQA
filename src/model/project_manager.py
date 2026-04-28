"""
Project Manager - Handles project.json CRUD operations
Manages project metadata, results storage, and project workflows
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

logger = logging.getLogger(__name__)

class ProjectManager:
    """
    Manages CALQA project workflows including creation, loading, and result persistence.

    Project folder layout (per project):
        <parent>/<project_name>/
            <project_name>.json
            training/
                output/
                    CPU_LoG/
                    CPU_DoG/
                    GPU_LoG/
                    GPU_DoG/

    Results are stored in the project JSON.
    """

    CURRENT_VERSION = "1.0"

    def __init__(self):
        self.current_project: Optional[Dict] = None
        self.project_path: Optional[Path] = None

    def _sanitize_name(self, name: str) -> str:
        """Sanitize project name to use in folder and file names."""
        return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name).strip("_")

    def create_project(
        self,
        name: str,
        description: str,
        project_type: str,
        parent_folder: str,
    ) -> bool:
        """Create a new project folder and initialize project JSON."""
        try:
            if not name or not name.strip():
                raise ValueError("Project name is required")

            project_name = self._sanitize_name(name.strip())
            parent_path = Path(parent_folder)
            parent_path.mkdir(parents=True, exist_ok=True)

            project_root = parent_path / project_name
            project_root.mkdir(parents=True, exist_ok=True)

            # Create required folder structure
            training_dir = project_root / "training"
            output_dir = training_dir / "output"
            output_dir.mkdir(parents=True, exist_ok=True)

            for method in ["CPU_LoG", "CPU_DoG", "GPU_LoG", "GPU_DoG"]:
                (output_dir / method).mkdir(parents=True, exist_ok=True)

            # Create project JSON file named after the project
            project_file = project_root / f"{project_name}.json"

            self.current_project = {
                "version": self.CURRENT_VERSION,
                "name": project_name,
                "description": description.strip(),
                "type": project_type,
                "folder": str(project_root),
                "created_at": datetime.now().isoformat(),
                "results": [],
            }

            self.project_path = project_file
            with open(self.project_path, "w") as f:
                json.dump(self.current_project, f, indent=2)

            return True
        except Exception as e:
            logger.error("Error creating project: %s", e)
            return False

    def load_project(self, project_json_path: str) -> bool:
        """Load an existing project JSON file."""
        try:
            json_file = Path(project_json_path)
            if not json_file.exists():
                raise FileNotFoundError(f"Project file not found: {project_json_path}")

            with open(json_file, "r") as f:
                self.current_project = json.load(f)

            self.project_path = json_file
            return True
        except Exception as e:
            logger.error("Error loading project: %s", e)
            return False

    def _get_project_root(self) -> Optional[Path]:
        if not self.current_project:
            return None
        return Path(self.current_project.get("folder", ""))

    def _get_output_folder(self, method: str) -> Optional[Path]:
        """Return output folder for the given method."""
        root = self._get_project_root()
        if not root:
            return None

        return root / "training" / "output" / method

    def get_output_path(self, method: str, input_file: str) -> Optional[Path]:
        """Return full output file path for a given input and method."""
        output_folder = self._get_output_folder(method)
        if not output_folder:
            return None

        output_folder.mkdir(parents=True, exist_ok=True)
        input_stem = Path(input_file).stem
        return output_folder / f"{input_stem}_{method}.png"

    def save_output_image(self, method: str, input_file: str, image_array) -> Optional[str]:
        """Save image array to output folder and return saved path."""
        try:
            import numpy as np
            
            output_path = self.get_output_path(method, input_file)
            if output_path is None:
                return None

            # Convert to uint8 for PNG compatibility
            if image_array.dtype == np.float32 or image_array.dtype == np.float64:
                # Normalize float values to 0-255
                if image_array.max() <= 1.0:
                    img_array = (image_array * 255).astype(np.uint8)
                else:
                    img_array = np.clip(image_array, 0, 255).astype(np.uint8)
            else:
                img_array = image_array.astype(np.uint8)

            img = Image.fromarray(img_array)
            img.save(output_path)
            return str(output_path)
        except Exception as e:
            logger.error("Error saving output image: %s", e)
            return None

    def _find_result_index(self, input_file: str, method: str) -> int:
        """Find index of an existing result by input file and method."""
        if not self.current_project:
            return -1
        results = self.current_project.get("results", [])
        for idx, r in enumerate(results):
            if r.get("input_file") == input_file and r.get("method") == method:
                return idx
        return -1

    def save_result(self, result_data: Dict) -> bool:
        """Add or update a result in the project JSON."""
        try:
            if not self.current_project:
                raise RuntimeError("No project loaded")

            # Add timestamp if not provided
            if "timestamp" not in result_data or not result_data["timestamp"]:
                result_data["timestamp"] = datetime.now().isoformat()

            # Avoid duplicate entries for same file+method
            idx = self._find_result_index(result_data.get("input_file", ""), result_data.get("method", ""))
            if idx >= 0:
                self.current_project["results"][idx].update(result_data)
            else:
                self.current_project["results"].append(result_data)

            # Save to disk
            with open(self.project_path, "w") as f:
                json.dump(self.current_project, f, indent=2)

            return True
        except Exception as e:
            logger.error("Error saving result: %s", e)
            return False

    def get_results(self) -> List[Dict]:
        """Get all results from current project."""
        if not self.current_project:
            return []
        return self.current_project.get("results", [])

    def get_project_info(self) -> Optional[Dict]:
        """Get current project metadata (without results)."""
        if not self.current_project:
            return None

        return {
            "name": self.current_project.get("name"),
            "description": self.current_project.get("description"),
            "type": self.current_project.get("type"),
            "folder": self.current_project.get("folder"),
            "created_at": self.current_project.get("created_at"),
        }

    def export_results_csv(self, output_path: str) -> bool:
        """Export all results as CSV file."""
        try:
            import csv

            if not self.current_project:
                raise RuntimeError("No project loaded")

            results = self.current_project.get("results", [])
            if not results:
                return False

            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)

            return True
        except Exception as e:
            logger.error("Error exporting CSV: %s", e)
            return False

    def export_results_json(self, output_path: str) -> bool:
        """Export all results as JSON file."""
        try:
            if not self.current_project:
                raise RuntimeError("No project loaded")

            results = self.current_project.get("results", [])
            if not results:
                return False

            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

            return True
        except Exception as e:
            logger.error("Error exporting JSON: %s", e)
            return False

    def delete_result(self, index: int) -> bool:
        """Delete a result by index."""
        try:
            if not self.current_project:
                raise RuntimeError("No project loaded")

            results = self.current_project.get("results", [])
            if 0 <= index < len(results):
                results.pop(index)

                # Save to disk
                with open(self.project_path, "w") as f:
                    json.dump(self.current_project, f, indent=2)

                return True
            return False
        except Exception as e:
            logger.error("Error deleting result: %s", e)
            return False

    def clear_results(self) -> bool:
        """Clear all results from current project."""
        try:
            if not self.current_project:
                raise RuntimeError("No project loaded")

            self.current_project["results"] = []

            # Save to disk
            with open(self.project_path, "w") as f:
                json.dump(self.current_project, f, indent=2)

            return True
        except Exception as e:
            logger.error("Error clearing results: %s", e)
            return False
