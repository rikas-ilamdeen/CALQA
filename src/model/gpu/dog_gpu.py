"""GPU DoG implementation for CALQA.

This module applies Difference of Gaussians (DoG) on grayscale images using
Numba CUDA kernels. The pipeline is:
1) Gaussian blur with sigma=1
2) Gaussian blur with sigma=2
3) Subtract the two outputs on GPU
4) Normalize result to [0, 1]

Gaussian(g1) - Gaussian(g2)
Used to highlights structure differences (edges or veins)
Faster and simpler than LoG
    1. Blurring the image twice (with different levels)
    2. Subtracting the two images

# high-performance separable-Gaussian DoG on CUDA with shared memory, pinned host arrays, batch support, and CPU fallback
"""

import time
import logging
import numpy as np
from numba import cuda, float32
from skimage.exposure import rescale_intensity
import warnings


logger = logging.getLogger(__name__)

# CPU fallback keeps the application usable on non-CUDA systems.
try:
    from model.cpu.dog_cpu import apply_dog as apply_dog_cpu
    CPU_FALLBACK_AVAILABLE = True
except ImportError:
    CPU_FALLBACK_AVAILABLE = False

# Thread block shape used by horizontal/vertical blur kernels.
BLOCK_X = 32
BLOCK_Y = 8
# Max convolution radius we expect (sigma=2 -> radius=6).
MAX_RADIUS = 6
# Shared-memory tile dimensions include halo pixels around each block.
SHARED_WIDTH = BLOCK_X + 2 * MAX_RADIUS
SHARED_HEIGHT = BLOCK_Y + 2 * MAX_RADIUS

# Stores GPU memory to reuse, instead of allocating memory every time
_gpu_buffers = {}

# This function creates a 1D Gaussian kernel used for blurring the image. Sigma controls how much blur is applied
def _make_gaussian_kernel(sigma):
    """Build a normalized 1D Gaussian kernel for separable convolution."""
    radius = int(3 * sigma)
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x**2) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    return kernel.astype(np.float32)

# We use two kernels because DoG requires two different blur levels
KERNEL1 = _make_gaussian_kernel(1.0)
KERNEL2 = _make_gaussian_kernel(2.0)
_RADIUS1 = (KERNEL1.size - 1) // 2
_RADIUS2 = (KERNEL2.size - 1) // 2

_kernel_cache = {}

def _get_device_kernel(kernel):
    """Transfer kernel to GPU once and reuse it in subsequent calls."""
    key = ('kernel', kernel.size)
    if key not in _kernel_cache:
        _kernel_cache[key] = cuda.to_device(kernel)
    return _kernel_cache[key]


def _get_gpu_buffers(batch_size, height, width):
    """Get or allocate reusable pinned-host and device buffers for a shape."""
    key = (batch_size, height, width)
    if key not in _gpu_buffers:
        # Pinned host memory speeds up host(CPU) and device(GPU) transfer.
        host_pinned = cuda.pinned_array((batch_size, height, width), dtype=np.float32)
        _gpu_buffers[key] = {
            'host': host_pinned,
            'd_input': cuda.device_array((batch_size, height, width), dtype=np.float32),
            'd_temp': cuda.device_array((batch_size, height, width), dtype=np.float32),
            'd_g1': cuda.device_array((batch_size, height, width), dtype=np.float32),
            'd_g2': cuda.device_array((batch_size, height, width), dtype=np.float32),
            'd_out': cuda.device_array((batch_size, height, width), dtype=np.float32),
        }
    return _gpu_buffers[key]

# Instead of applying a full 2D Gaussian filter
# split it into horizontal and vertical passes
# This reduces computation complexity and improves performance
# Applies Gaussian blur row-wise, Uses shared memory (fast GPU memory)
@cuda.jit
def _horizontal_blur(d_in, d_out, height, width, batch_size, kernel, radius):
    """Apply one horizontal Gaussian pass using shared-memory tiling."""
    # Shared tile stores one block row plus left/right halo.
    shared = cuda.shared.array((SHARED_HEIGHT, SHARED_WIDTH), dtype=float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bz = cuda.blockIdx.z

    bdx = cuda.blockDim.x
    bdy = cuda.blockDim.y

    x = bx * bdx + tx
    y = by * bdy + ty

    if bz >= batch_size or y >= height:
        return

    in_img = d_in[bz]
    out_img = d_out[bz]

    # Global x-index where this block's shared tile starts (includes halo).
    tile_start = bx * bdx - radius
    tile_width = bdx + 2 * radius

    # Cooperative load: each thread loads multiple elements if needed.
    for i in range(tx, tile_width, bdx):
        gx = tile_start + i
        if 0 <= gx < width:
            shared[ty, i] = in_img[y, gx]
        else:
            # Zero padding outside image bounds.
            shared[ty, i] = 0.0

    cuda.syncthreads()

    # Convolve current pixel using cached shared-memory neighborhood.
    if x < width and y < height:
        acc = 0.0
        for k in range(2 * radius + 1):
            acc += kernel[k] * shared[ty, tx + k]
        out_img[y, x] = acc

# Applies Gaussian blur column-wise
@cuda.jit
def _vertical_blur(d_in, d_out, height, width, batch_size, kernel, radius):
    """Apply one vertical Gaussian pass using shared-memory tiling."""
    # Shared tile stores one block column plus top/bottom halo.
    shared = cuda.shared.array((SHARED_HEIGHT, SHARED_WIDTH), dtype=float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bz = cuda.blockIdx.z

    bdx = cuda.blockDim.x
    bdy = cuda.blockDim.y

    x = bx * bdx + tx
    y = by * bdy + ty

    if bz >= batch_size or x >= width:
        return

    in_img = d_in[bz]
    out_img = d_out[bz]

    # Global y-index where this block's shared tile starts (includes halo).
    tile_start = by * bdy - radius
    tile_height = bdy + 2 * radius

    # Cooperative load: each thread loads multiple elements if needed.
    for i in range(ty, tile_height, bdy):
        gy = tile_start + i
        if 0 <= gy < height:
            shared[i, tx] = in_img[gy, x]
        else:
            # Zero padding outside image bounds.
            shared[i, tx] = 0.0

    cuda.syncthreads()

    # Convolve current pixel using cached shared-memory neighborhood.
    if x < width and y < height:
        acc = 0.0
        for k in range(2 * radius + 1):
            acc += kernel[k] * shared[ty + k, tx]
        out_img[y, x] = acc

# This kernel subtracts the two blurred images pixel by pixel to produce the DoG result
@cuda.jit
def _subtract_batch(d_g1, d_g2, d_out, height, width, batch_size):
    """Element-wise subtraction: DoG = blur(sigma=1) - blur(sigma=2)."""
    x, y, bz = cuda.grid(3)
    if bz < batch_size and x < width and y < height:
        d_out[bz, y, x] = d_g1[bz, y, x] - d_g2[bz, y, x]

# First GPU run is slow (compilation)
def _warmup_gpu():
    """Compile kernels once with a realistic launch size to reduce first-run latency."""
    try:
        # Use a larger warm-up launch so Numba doesn't flag low-occupancy grids.
        # This compiles kernels once at startup with realistic image dimensions.
        h, w, batch = 256, 256, 8
        dummy = np.zeros((batch, h, w), dtype=np.float32)
        d_in = cuda.to_device(dummy)
        d_tmp = cuda.device_array_like(d_in)
        d_g1 = cuda.device_array_like(d_in)
        d_g2 = cuda.device_array_like(d_in)
        d_out = cuda.device_array_like(d_in)

        blocks = ((w + BLOCK_X - 1) // BLOCK_X, (h + BLOCK_Y - 1) // BLOCK_Y, batch)
        threads = (BLOCK_X, BLOCK_Y)

        d_kernel1 = _get_device_kernel(KERNEL1)
        d_kernel2 = _get_device_kernel(KERNEL2)
        _horizontal_blur[blocks, threads](d_in, d_tmp, h, w, batch, d_kernel1, _RADIUS1)
        _vertical_blur[blocks, threads](d_tmp, d_g1, h, w, batch, d_kernel1, _RADIUS1)
        _horizontal_blur[blocks, threads](d_in, d_tmp, h, w, batch, d_kernel2, _RADIUS2)
        _vertical_blur[blocks, threads](d_tmp, d_g2, h, w, batch, d_kernel2, _RADIUS2)
        _subtract_batch[blocks, (BLOCK_X, BLOCK_Y, 1)](d_g1, d_g2, d_out, h, w, batch)
        cuda.synchronize()
    except Exception as e:
        # Warm-up is optional; keep startup resilient if CUDA context is unavailable.
        logger.debug("Skipping DoG GPU warm-up: %s", e)


_warmup_gpu()


def apply_dog(image):
    """Run DoG on GPU for a single 2D image or a 3D image batch.

    Args:
        image: 2D grayscale image or 3D stack (batch, height, width).

    Returns:
        tuple: (result_image_or_batch, kernel_time_ms)
    """
    try:
        # The function supports both single image and batch processing
        if image.ndim == 2:
            batch_input = image[np.newaxis, ...].astype(np.float32)
        elif image.ndim == 3:
            batch_input = image.astype(np.float32)
        else:
            raise ValueError("Input image must be 2D or 3D grayscale stack")

        # Reuses pre-allocated memory to improve performance
        batch_size, height, width = batch_input.shape
        buffers = _get_gpu_buffers(batch_size, height, width)
        host_pinned = buffers['host']
        d_input = buffers['d_input']
        d_temp = buffers['d_temp']
        d_g1 = buffers['d_g1']
        d_g2 = buffers['d_g2']
        d_out = buffers['d_out']

        # Host -> Device transfer (pinned host memory already prepared).
        np.copyto(host_pinned, batch_input)
        d_input.copy_to_device(host_pinned)

        # 3D launch: x, y for pixels and z for batch index.
        blocks = ((width + BLOCK_X - 1) // BLOCK_X, (height + BLOCK_Y - 1) // BLOCK_Y, batch_size)
        threads = (BLOCK_X, BLOCK_Y)

        start = time.perf_counter()

        d_kernel1 = _get_device_kernel(KERNEL1)
        d_kernel2 = _get_device_kernel(KERNEL2)

        # Kernel Execution: Each pixel is processed in parallel across thousands of GPU threads
        # Two separable Gaussian passes per sigma, then subtraction
        _horizontal_blur[blocks, threads](d_input, d_temp, height, width, batch_size, d_kernel1, _RADIUS1)
        _vertical_blur[blocks, threads](d_temp, d_g1, height, width, batch_size, d_kernel1, _RADIUS1)
        _horizontal_blur[blocks, threads](d_input, d_temp, height, width, batch_size, d_kernel2, _RADIUS2)
        _vertical_blur[blocks, threads](d_temp, d_g2, height, width, batch_size, d_kernel2, _RADIUS2)
        _subtract_batch[blocks, (BLOCK_X, BLOCK_Y, 1)](d_g1, d_g2, d_out, height, width, batch_size)

        # Synchronization: Ensures all GPU operations are completed before measuring time
        cuda.synchronize()

        # Measures GPU execution time for benchmarking
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        # Device -> Host transfer into pinned buffer (Copy Back to CPU)
        d_out.copy_to_host(host_pinned)

        # Rescale each output image for consistent display/analysis.
        result = np.empty_like(host_pinned)
        for i in range(batch_size):
            result[i] = rescale_intensity(host_pinned[i], in_range='image', out_range=(0, 1))

        # Returns processed image and execution time
        final_result = result[0] if image.ndim == 2 else result
        return (final_result, elapsed_ms)

    # If GPU fails, system automatically switches to CPU version, ensuring reliability
    except Exception as e:
        warnings.warn(f"GPU processing failed ({e}), falling back to CPU", UserWarning)
        if CPU_FALLBACK_AVAILABLE:
            if image.ndim == 2:
                return (apply_dog_cpu(image), None)
            else:
                return (np.stack([apply_dog_cpu(img) for img in image]), None)
        else:
            raise RuntimeError(f"GPU processing failed and no CPU fallback available: {e}")
