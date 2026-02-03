# CALQA – Centella Asiatica Leaves Quality Assessment

CALQA (Centella Asiatica Leaves Quality Assessment) is a GPU-accelerated image segmentation system for automated quality analysis of Gotu Kola leaves using Difference of Gaussian (DoG) and CUDA. The system focuses on leaf vein enhancement and segmentation, which is a critical step for assessing leaf health and quality.

## System Architecture

The project follows the **Model–View–Controller (MVC)** architecture.

### Model
- Image loading
- Preprocessing (grayscale, Gaussian blur)
- CPU implementations of LoG and DoG
- Performance timing

### View
- Tkinter-based GUI
- Image preview (original and processed)
- Filter selection
- Processing time display

### Controller
- Handles user actions
- Connects GUI with processing logic
- Manages the full processing pipeline
