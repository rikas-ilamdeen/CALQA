# CALQA – Centella Asiatica Leaves Quality Assessment

CALQA (Centella Asiatica Leaves Quality Assessment) is a desktop application for GPU-accelerated image segmentation and quality assessment of Centella Asiatica leaves, also known as Gotu Kola. The project uses image processing techniques to highlight leaf vein structures, compare CPU and GPU execution, and support batch analysis through a clean PyQt6 interface.

## Project Overview

Leaf quality assessment is often done manually, which is slow, subjective, and difficult to standardize. CALQA addresses this problem by automating the segmentation and analysis pipeline using image filtering, grayscale preprocessing, and CUDA-based GPU acceleration.

The system is useful for:

- automated inspection of medicinal plant leaves
- faster quality grading in agricultural workflows
- experimental comparison of CPU and GPU image processing
- university-level research in computer vision and parallel computing

## Problem Statement

Manual leaf inspection depends on human judgment and can vary from person to person. For large batches of images, this becomes time-consuming and inefficient. CALQA aims to reduce that burden by:

- detecting vein-like structures automatically
- processing images faster using GPU parallelism
- providing measurable timing and comparison results
- storing results inside a project-based workflow

## Key Objectives

- Build a user-friendly desktop application for leaf image analysis
- Implement both CPU and GPU versions of DoG and LoG filtering
- Improve processing speed using CUDA parallel computation
- Provide batch processing for multiple images
- Save project data and processing results in structured folders
- Support accuracy validation using SSIM

## Key Features

- PyQt6-based graphical user interface
- Single-image segmentation workflow
- Batch processing of folders containing images
- CPU and GPU algorithm selection
- DoG and LoG filtering support
- Feature extraction for vein density estimation
- SSIM-based validation for comparing outputs
- Project management using JSON metadata
- Results table and benchmark comparison tab
- GPU buffer caching and pinned memory for better performance

## System Architecture

CALQA follows the **Model–View–Controller (MVC)** pattern.

### 1. Model Layer
The model layer contains the core data and processing logic.

It includes:

- image loading
- grayscale conversion and preprocessing
- CPU-based DoG and LoG filtering
- GPU-based DoG and LoG filtering using Numba CUDA
- feature extraction for vein density
- SSIM validation
- project storage and result management
- timing utilities

### 2. View Layer
The view layer contains the user interface.

It includes:

- welcome screen
- new project dialog
- main PyQt6 window
- segmentation tab
- batch processing tab
- results tab
- benchmark tab

### 3. Controller Layer
The controller layer connects the GUI to the model logic.

It handles:

- loading and opening projects
- user actions from the interface
- image processing requests
- batch processing flow
- result saving and display updates
- SSIM validation workflow

## System Workflow

The application works in the following order:

1. The user opens CALQA.
2. A project is created or an existing one is loaded.
3. The main PyQt6 window opens.
4. The user loads a leaf image or a folder of images.
5. The user selects CPU or GPU mode and chooses DoG or LoG.
6. The image is converted to grayscale.
7. The selected filter is applied.
8. The result is displayed and timing is measured.
9. Features such as vein density may be extracted.
10. Output images and metadata are saved inside the project folder.

## Technologies Used

### Core Technologies

- **Python** - main programming language
- **PyQt6** - desktop GUI framework
- **NumPy** - numerical array operations
- **SciPy** - CPU convolution and image filtering support
- **scikit-image** - image loading, grayscale conversion, and intensity rescaling
- **Pillow** - saving processed images and image file handling

### GPU and Parallel Computing

- **Numba CUDA** - GPU kernel programming and accelerated execution
- CUDA shared memory and device arrays for optimized processing
- pinned host memory for reducing transfer overhead

### Supporting Tools

- JSON for project metadata
- SSIM from scikit-image for accuracy comparison
- Git and PyInstaller for development and packaging

## GPU vs CPU Explanation

CALQA supports both CPU and GPU execution so that results can be compared fairly.

- **CPU mode** runs the image processing algorithms using standard Python libraries and SciPy/scikit-image functions.
- **GPU mode** runs the same conceptual processing using CUDA kernels through Numba, which allows parallel processing of image pixels.

The CPU version is useful as a baseline and fallback. The GPU version is designed for faster execution, especially when processing larger images or multiple images.

## Installation

### Prerequisites

- Python 3.11 or compatible version
- NVIDIA GPU with CUDA support for GPU mode
- CUDA-capable driver installed
- Virtual environment recommended

### Steps

```bash
git clone https://github.com/rikas-ilamdeen/CALQA.git
cd CALQA
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

If you only want to run the CPU version, the project may still start without a CUDA-capable GPU, but GPU features will not work.

## How to Run the Project

Run the application from the project root:

```bash
python src/main.py
```

If you build the standalone executable with PyInstaller, run the generated `.exe` file from the `dist` folder instead.

## Example Usage Workflow

### Single Image Segmentation

1. Launch the application.
2. Create a new project or open an existing one.
3. Open the Segmentation tab.
4. Click **Load Image** and choose a leaf image.
5. Select **CPU** or **GPU**.
6. Select **LoG** or **DoG**.
7. Click **Compute**.
8. View the processed result and processing time.

### Batch Processing

1. Open the Batch Processing tab.
2. Select a folder containing multiple images.
3. Choose the processing method.
4. Click **Process All**.
5. Review the batch results table and timing summary.

### Results and Benchmarking

1. Open the Results tab to view saved outputs.
2. Export results as CSV or JSON if needed.
3. Use the Benchmark tab to compare CPU and GPU methods.
4. Use SSIM validation to check output similarity where applicable.

## Project Folder Structure

```text
CALQA/
├── README.md
├── requirements.txt
├── CALQA.spec
├── data/
│   ├── samples/
│   └── centella-asiatica-leaf-image-dataset/
├── src/
│   ├── main.py
│   ├── controller/
│   │   ├── app_controller.py
│   │   └── project_controller.py
│   ├── model/
│   │   ├── feature_extractor.py
│   │   ├── image_loader.py
│   │   ├── preprocessing.py
│   │   ├── project_manager.py
│   │   ├── ssim_validator.py
│   │   ├── timer.py
│   │   ├── cpu/
│   │   │   ├── dog_cpu.py
│   │   │   └── log_cpu.py
│   │   └── gpu/
│   │       ├── dog_gpu.py
│   │       └── log_gpu.py
│   └── view/
│       ├── main_window_pyqt.py
│       ├── new_project_dialog.py
│       ├── themes.py
│       └── welcome_screen.py
└── venv/
```

## Sample Output

The application produces:

- processed grayscale or enhanced segmentation images
- processing time in milliseconds
- saved output images inside the project folder
- project metadata stored in JSON format
- batch tables with file name, method, time, and status
- benchmark comparison results

In practical terms, a successful output shows the detected vein or filter response of the leaf image, which can then be used for further quality analysis.

## Performance and Validation

The project includes both performance comparison and output validation.

- **Performance comparison**: CPU vs GPU timing is measured to show the benefit of parallel processing.
- **Accuracy validation**: SSIM is used to compare outputs where CPU and GPU results should be similar.

This makes the project useful not only for implementation, but also for reporting experimental results in an academic setting.

## Future Improvements

- add more advanced segmentation methods
- include automatic classification of leaf quality grades
- improve benchmark reporting with graphs and charts
- support more image formats and larger datasets
- add exporting to PDF reports
- improve CUDA optimization for larger batch workloads
- add model-based learning or deep learning integration later

## Notes

- The project uses a PyQt6-based interface, not Tkinter.
- GPU features require a CUDA-capable NVIDIA GPU and compatible drivers.
