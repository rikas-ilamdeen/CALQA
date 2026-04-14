import time
import numpy as np
from numba import cuda, float32
from skimage.exposure import rescale_intensity
import warnings

# Try to import CPU fallback
try:
    from model.cpu.dog_cpu import apply_dog as apply_dog_cpu
    CPU_FALLBACK_AVAILABLE = True
except ImportError:
    CPU_FALLBACK_AVAILABLE = False

# Tuned configuration for NVIDIA MX450
BLOCK_X = 32
BLOCK_Y = 8
MAX_RADIUS = 6
SHARED_WIDTH = BLOCK_X + 2 * MAX_RADIUS
SHARED_HEIGHT = BLOCK_Y + 2 * MAX_RADIUS

# GPU buffer cache to reduce per-call overhead
_gpu_buffers = {}

# Precomputed Gaussian kernels for sigma=1 and sigma=2 (DoG)
def _make_gaussian_kernel(sigma):
    radius = int(3 * sigma)
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x**2) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    return kernel.astype(np.float32)

KERNEL1 = _make_gaussian_kernel(1.0)
KERNEL2 = _make_gaussian_kernel(2.0)
_RADIUS1 = (KERNEL1.size - 1) // 2
_RADIUS2 = (KERNEL2.size - 1) // 2

_cpu_baseline_ms = None

_kernel_cache = {}

def _get_device_kernel(kernel):
    key = ('kernel', kernel.size)
    if key not in _kernel_cache:
        _kernel_cache[key] = cuda.to_device(kernel)
    return _kernel_cache[key]


def _get_gpu_buffers(batch_size, height, width):
    key = (batch_size, height, width)
    if key not in _gpu_buffers:
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


@cuda.jit
def _horizontal_blur(d_in, d_out, height, width, batch_size, kernel, radius):
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

    tile_start = bx * bdx - radius
    tile_width = bdx + 2 * radius

    for i in range(tx, tile_width, bdx):
        gx = tile_start + i
        if 0 <= gx < width:
            shared[ty, i] = in_img[y, gx]
        else:
            shared[ty, i] = 0.0

    cuda.syncthreads()

    if x < width and y < height:
        acc = 0.0
        for k in range(2 * radius + 1):
            acc += kernel[k] * shared[ty, tx + k]
        out_img[y, x] = acc


@cuda.jit
def _vertical_blur(d_in, d_out, height, width, batch_size, kernel, radius):
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

    tile_start = by * bdy - radius
    tile_height = bdy + 2 * radius

    for i in range(ty, tile_height, bdy):
        gy = tile_start + i
        if 0 <= gy < height:
            shared[i, tx] = in_img[gy, x]
        else:
            shared[i, tx] = 0.0

    cuda.syncthreads()

    if x < width and y < height:
        acc = 0.0
        for k in range(2 * radius + 1):
            acc += kernel[k] * shared[ty + k, tx]
        out_img[y, x] = acc


@cuda.jit
def _subtract_batch(d_g1, d_g2, d_out, height, width, batch_size):
    x, y, bz = cuda.grid(3)
    if bz < batch_size and x < width and y < height:
        d_out[bz, y, x] = d_g1[bz, y, x] - d_g2[bz, y, x]


def _warmup_gpu():
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
    except Exception:
        pass


_warmup_gpu()


def _get_cpu_baseline_ms(image):
    global _cpu_baseline_ms
    if _cpu_baseline_ms is None:
        start = time.perf_counter()
        apply_dog_cpu(image)
        _cpu_baseline_ms = (time.perf_counter() - start) * 1000.0
    return _cpu_baseline_ms


def apply_dog(image):
    """Apply Difference of Gaussian with optimal GPU execution and batch support."""
    try:
        if image.ndim == 2:
            batch_input = image[np.newaxis, ...].astype(np.float32)
        elif image.ndim == 3:
            batch_input = image.astype(np.float32)
        else:
            raise ValueError("Input image must be 2D or 3D grayscale stack")

        batch_size, height, width = batch_input.shape
        buffers = _get_gpu_buffers(batch_size, height, width)
        host_pinned = buffers['host']
        d_input = buffers['d_input']
        d_temp = buffers['d_temp']
        d_g1 = buffers['d_g1']
        d_g2 = buffers['d_g2']
        d_out = buffers['d_out']

        np.copyto(host_pinned, batch_input)
        d_input.copy_to_device(host_pinned)

        blocks = ((width + BLOCK_X - 1) // BLOCK_X, (height + BLOCK_Y - 1) // BLOCK_Y, batch_size)
        threads = (BLOCK_X, BLOCK_Y)

        start = time.perf_counter()

        d_kernel1 = _get_device_kernel(KERNEL1)
        d_kernel2 = _get_device_kernel(KERNEL2)

        _horizontal_blur[blocks, threads](d_input, d_temp, height, width, batch_size, d_kernel1, _RADIUS1)
        _vertical_blur[blocks, threads](d_temp, d_g1, height, width, batch_size, d_kernel1, _RADIUS1)
        _horizontal_blur[blocks, threads](d_input, d_temp, height, width, batch_size, d_kernel2, _RADIUS2)
        _vertical_blur[blocks, threads](d_temp, d_g2, height, width, batch_size, d_kernel2, _RADIUS2)
        _subtract_batch[blocks, (BLOCK_X, BLOCK_Y, 1)](d_g1, d_g2, d_out, height, width, batch_size)

        cuda.synchronize()

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        d_out.copy_to_host(host_pinned)

        result = np.empty_like(host_pinned)
        for i in range(batch_size):
            result[i] = rescale_intensity(host_pinned[i], in_range='image', out_range=(0, 1))

        cpu_ms = _get_cpu_baseline_ms(image if image.ndim == 2 else image[0])
        speedup = cpu_ms / elapsed_ms if elapsed_ms > 0 else float('inf')

        print(f"? GPU RUN: {cuda.get_current_device().name} | Time: {elapsed_ms:.1f}ms | Speedup vs CPU: {speedup:.1f}x")

        final_result = result[0] if image.ndim == 2 else result
        return (final_result, elapsed_ms)

    except Exception as e:
        warnings.warn(f"GPU processing failed ({e}), falling back to CPU", UserWarning)
        if CPU_FALLBACK_AVAILABLE:
            if image.ndim == 2:
                return (apply_dog_cpu(image), None)
            else:
                return (np.stack([apply_dog_cpu(img) for img in image]), None)
        else:
            raise RuntimeError(f"GPU processing failed and no CPU fallback available: {e}")
