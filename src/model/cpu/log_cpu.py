"""CPU baseline implementation of Laplacian of Gaussian (LoG).

This module provides a mathematically explicit LoG pipeline that is useful
for explaining the algorithm in viva and for comparing with GPU output.
"""

import numpy as np
from scipy import ndimage
from skimage.exposure import rescale_intensity


def log_kernel(size, sigma):
    """Generate an analytic 2D LoG kernel.

    Args:
        size: Kernel width/height (odd integer)
        sigma: Standard deviation of Gaussian component

    Returns:
        2D numpy array containing LoG filter weights
    """
    kernel = np.zeros((size, size))
    center = size // 2

    # Compute LoG value for each kernel position relative to center.
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            r2 = x**2 + y**2
            kernel[i, j] = - (1 / (np.pi * sigma**4)) * (1 - r2 / (2 * sigma**2)) * np.exp(-r2 / (2 * sigma**2))
    return kernel


def apply_log(gray):
    """Apply Laplacian of Gaussian filtering on CPU.

    Uses direct 2D convolution with an analytically derived LoG kernel.

    Args:
        gray: 2D grayscale image

    Returns:
        2D float image normalized to [0, 1]
    """
    # Sigma=1.4 is a common LoG choice and approximates DoG(1,2).
    sigma = 1.4
    kernel_size = 2 * int(3 * sigma) + 1  # 9 for sigma=1.4

    kernel = log_kernel(kernel_size, sigma)
    log_img = ndimage.convolve(gray.astype(float), kernel)

    # Normalize result for consistent visualization and comparison.
    log_img = rescale_intensity(log_img, in_range='image', out_range=(0, 1))
    return log_img