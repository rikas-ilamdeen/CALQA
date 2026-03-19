import time
import numpy as np
from numba import cuda
from skimage.exposure import rescale_intensity
import warnings

# Try to import CPU fallback
try:
    from model.cpu.log_cpu import apply_log as apply_log_cpu
    CPU_FALLBACK_AVAILABLE = True
except ImportError:
    CPU_FALLBACK_AVAILABLE = False

# single LoG kernel computed on CPU and cached
def _build_log_kernel(size=9, sigma=1.4):
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

# Cached device kernel to avoid repeated transfers
_log_kernel_device = None

_cpu_baseline_ms = None


def _get_log_kernel_device():
    global _log_kernel_device
    if _log_kernel_device is None:
        _log_kernel_device = cuda.to_device(LOG_KERNEL)
    return _log_kernel_device


def _get_cpu_baseline_ms(gray):
    global _cpu_baseline_ms
    if _cpu_baseline_ms is None:
        start = time.perf_counter()
        apply_log_cpu(gray)
        _cpu_baseline_ms = (time.perf_counter() - start) * 1000.0
    return _cpu_baseline_ms


@cuda.jit
def _convolve_2d(image, kernel, output, height, width):
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
        output[y, x] = acc


def apply_log(gray):
    try:
        if gray.ndim != 2:
            raise ValueError('Input image must be 2D grayscale')

        height, width = gray.shape
        host_pinned = cuda.pinned_array((height, width), dtype=np.float32)
        np.copyto(host_pinned, gray.astype(np.float32))

        d_image = cuda.to_device(host_pinned)
        d_output = cuda.device_array((height, width), dtype=np.float32)
        d_kernel = _get_log_kernel_device()

        threads = (16, 16)
        blocks = ((width + threads[0] - 1) // threads[0], (height + threads[1] - 1) // threads[1])

        start = time.perf_counter()
        _convolve_2d[blocks, threads](d_image, d_kernel, d_output, height, width)
        cuda.synchronize()

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        d_output.copy_to_host(host_pinned)

        cpu_ms = _get_cpu_baseline_ms(gray)
        speedup = cpu_ms / elapsed_ms if elapsed_ms > 0 else float('inf')
        print(f"[GPU LoG] Time: {elapsed_ms:.1f}ms | Speedup vs CPU: {speedup:.1f}x")

        result = rescale_intensity(host_pinned, in_range='image', out_range=(0, 1))
        return (result, elapsed_ms)

    except Exception as e:
        warnings.warn(f"GPU processing failed ({e}), falling back to CPU", UserWarning)
        if CPU_FALLBACK_AVAILABLE:
            return (apply_log_cpu(gray), None)
        else:
            raise RuntimeError(f"GPU processing failed and no CPU fallback available: {e}")
