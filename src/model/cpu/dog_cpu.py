"""CPU baseline implementation of Difference of Gaussians (DoG).

This module is used as:
1) a reference output for algorithm validation,
2) a fallback when GPU execution is unavailable.

Gaussian(g1) - Gaussian(g2)
Used to highlights structure differences (edges or veins)
Faster and simpler than LoG
    1. Blurring the image twice (with different levels)
    2. Subtracting the two images
"""

from skimage.filters import gaussian
from skimage.exposure import rescale_intensity


def apply_dog(image):
    """Apply Difference of Gaussians on a grayscale image.

    Args:
        image: 2D numpy array (grayscale image)

    Returns:
        2D float image normalized to [0, 1]
    """
    # Two Gaussian smoothings at different scales.
    g1 = gaussian(image, sigma=1)
    g2 = gaussian(image, sigma=2)

    # DoG emphasizes structures that differ between the two scales.
    dog = g1 - g2

    # Normalize output range for consistent display and downstream metrics.
    dog = rescale_intensity(dog, in_range='image', out_range=(0, 1))
    return dog