"""GPU LoG implementation for CALQA.

This module applies Laplacian of Gaussian (LoG) on a 2D grayscale image
using a CUDA convolution kernel with a precomputed 9x9 LoG filter.
"""

import time
import logging
import numpy as np
from numba import cuda
from skimage.exposure import rescale_intensity
import warnings


logger = logging.getLogger(__name__)

# CPU fallback keeps the application usable on systems without CUDA support.
try:
    from model.cpu.log_cpu import apply_log as apply_log_cpu
    CPU_FALLBACK_AVAILABLE = True
except ImportError:
    CPU_FALLBACK_AVAILABLE = False

# Single LoG kernel computed on CPU and cached for reuse.
def _build_log_kernel(size=9, sigma=1.4):
    """Build analytic LoG kernel and center it to near-zero mean."""
    center = size // 2
    k = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            r2 = x * x + y * y
            k[i, j] = - (1.0 / (np.pi * sigma**4)) * (1 - r2 / (2 * sigma**2)) * np.exp(-r2 / (2 * sigma**2))
    k -= k.mean()
    return k

LOG_KERNEL = _build_log_kernel(9, 1.4)

# Cached device kernel to avoid repeated host->device transfers.
_log_kernel_device = None


def _get_log_kernel_device():
    """Transfer LoG kernel to GPU once and reuse it across calls."""
    global _log_kernel_device
    if _log_kernel_device is None:
        _log_kernel_device = cuda.to_device(LOG_KERNEL)
    return _log_kernel_device


@cuda.jit
def _convolve_2d(image, kernel, output, height, width):
    """Direct 2D convolution kernel for LoG filtering."""
    x, y = cuda.grid(2)
    size = 9
    center = 4
    if x < width and y < height:
        acc = 0.0
        for i in range(size):
            for j in range(size):
                yi = y + i - center
                xi = x + j - center
                if 0 <= yi < height and 0 <= xi < width:
                    acc += image[yi, xi] * kernel[i, j]
                # Out-of-bounds values are treated as zero padding.
        output[y, x] = acc


def apply_log(gray):
    """Apply GPU-accelerated Laplacian of Gaussian on a 2D grayscale image.

    Returns:
        tuple: (result_image, kernel_time_ms)
    """
    try:
        if gray.ndim != 2:
            raise ValueError('Input image must be 2D grayscale')

        # Allocate pinned host memory to improve transfer efficiency.
        height, width = gray.shape
        host_pinned = cuda.pinned_array((height, width), dtype=np.float32)
        np.copyto(host_pinned, gray.astype(np.float32))

        # Allocate device input/output arrays and fetch cached LoG kernel.
        d_image = cuda.to_device(host_pinned)
        d_output = cuda.device_array((height, width), dtype=np.float32)
        d_kernel = _get_log_kernel_device()

        # 2D launch grid: one thread computes one output pixel.
        threads = (16, 16)
        blocks = ((width + threads[0] - 1) // threads[0], (height + threads[1] - 1) // threads[1])

        # Measure kernel-only time for fair CPU/GPU comparison in UI.
        start = time.perf_counter()
        _convolve_2d[blocks, threads](d_image, d_kernel, d_output, height, width)
        cuda.synchronize()

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        # Copy result back to host and normalize for display.
        d_output.copy_to_host(host_pinned)

        result = rescale_intensity(host_pinned, in_range='image', out_range=(0, 1))
        return (result, elapsed_ms)

    except Exception as e:
        warnings.warn(f"GPU processing failed ({e}), falling back to CPU", UserWarning)
        if CPU_FALLBACK_AVAILABLE:
            logger.debug("LoG GPU fallback to CPU due to: %s", e)
            return (apply_log_cpu(gray), None)
        else:
            raise RuntimeError(f"GPU processing failed and no CPU fallback available: {e}")
