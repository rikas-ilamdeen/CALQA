"""
Feature Extractor - Extracts vein density features from a segmented leaf image.

Operates on the DoG/LoG segmented output (float [0,1] or uint8 [0,255]).
All computation uses numpy only.
"""

import numpy as np


def extract(segmented_image) -> dict:
    """
    Extract vein features from a segmented (grayscale) image.

    Vein density is computed on the segmented output, where bright pixels
    represent detected vein structures from DoG/LoG processing.

    Args:
        segmented_image: numpy array (float [0,1] or uint8 [0,255])

    Returns:
        dict with keys:
            - vein_density  (float): percentage of vein pixels [0.0 – 100.0]
            - edge_count    (int):   raw number of vein pixels detected
    """
    if segmented_image is None:
        return {"vein_density": 0.0, "edge_count": 0}

    img = segmented_image.astype(np.float64)

    # Normalise uint8 [0-255] → float [0-1]
    if img.max() > 1.0:
        img = img / 255.0

    total_pixels = img.size
    if total_pixels == 0:
        return {"vein_density": 0.0, "edge_count": 0}

    # Count pixels above 0.5 threshold as "vein/edge detected"
    bright_pixels = int(np.sum(img > 0.5))

    vein_density = round((bright_pixels / total_pixels) * 100, 2)

    return {
        "vein_density": vein_density,
        "edge_count": bright_pixels,
    }
