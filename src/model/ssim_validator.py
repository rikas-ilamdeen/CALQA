"""
SSIM Validator - Structural Similarity Index between two segmented images
Used to validate that GPU DoG output is numerically equivalent to CPU DoG output.
"""

import numpy as np


def compute_ssim(img_ref, img_cmp) -> float | None:
    """
    Compute Structural Similarity Index (SSIM) between two images.

    Both inputs are normalised to float [0, 1] before comparison,
    so the caller does not need to pre-process them.

    Args:
        img_ref: Reference image (numpy array, any numeric dtype)
        img_cmp: Comparison image (numpy array, any numeric dtype)

    Returns:
        SSIM score as float in [-1, 1] (1.0 = identical),
        or None if shapes don't match or an error occurs.
    """
    try:
        from skimage.metrics import structural_similarity as ssim

        if img_ref is None or img_cmp is None:
            return None

        # Must be same shape
        if img_ref.shape != img_cmp.shape:
            return None

        # Normalise to [0, 1] float
        ref = img_ref.astype(np.float64)
        cmp = img_cmp.astype(np.float64)

        if ref.max() > 1.0:
            ref = ref / 255.0
        if cmp.max() > 1.0:
            cmp = cmp / 255.0

        score = ssim(ref, cmp, data_range=1.0)
        return round(float(score), 4)

    except Exception as e:
        print(f"[SSIM] Error computing SSIM: {e}")
        return None
