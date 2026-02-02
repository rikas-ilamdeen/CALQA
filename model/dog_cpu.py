import time
from skimage import filters, color, img_as_float

def dog_cpu(image):
    """
    Apply Difference of Gaussian (DoG) on CPU
    """
    start_time = time.perf_counter()

    # Convert to grayscale if RGB
    if image.ndim == 3:
        image = color.rgb2gray(image)

    image = img_as_float(image)

    sigma1 = 1.0
    sigma2 = 2.0

    blur1 = filters.gaussian(image, sigma=sigma1)
    blur2 = filters.gaussian(image, sigma=sigma2)

    dog = blur1 - blur2

    end_time = time.perf_counter()
    time_ms = (end_time - start_time) * 1000

    return dog, time_ms
