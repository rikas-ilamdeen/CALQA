import numpy as np
from scipy import ndimage
from skimage.exposure import rescale_intensity


def log_kernel(size, sigma):
    """
    Generate Laplacian of Gaussian (LoG) 2D kernel
    LoG implementation based on Marr & Hildreth (1980) and HIPR2 tutorial
    """
    kernel = np.zeros((size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            r2 = x**2 + y**2
            kernel[i, j] = - (1 / (np.pi * sigma**4)) * (1 - r2 / (2 * sigma**2)) * np.exp(-r2 / (2 * sigma**2))
    return kernel


def apply_log(gray):
    """    
    Apply Laplacian of Gaussian (LoG)
    Uses direct 2D convolution with analytically derived LoG kernel.
    """
    sigma = 1.4                  # (common value: 1.0-1.6) approximating DoG(1,2)
    kernel_size = 2 * int(3 * sigma) + 1  # 9 for sigma=1.4

    kernel = log_kernel(kernel_size, sigma)
    log_img = ndimage.convolve(gray.astype(float), kernel)

    # Normalize for visualization (zero-crossings become visible)
    log_img = rescale_intensity(log_img, in_range='image', out_range=(0, 1))
    return log_img