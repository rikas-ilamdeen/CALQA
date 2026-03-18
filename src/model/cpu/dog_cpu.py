from skimage.filters import gaussian
from skimage.exposure import rescale_intensity

def apply_dog(image):
    """
    Apply Difference of Gaussians (DoG)
    """
    g1 = gaussian(image, sigma=1)
    g2 = gaussian(image, sigma=2)
    dog = g1 - g2
    # Normalize for visible veins
    dog = rescale_intensity(dog, in_range='image', out_range=(0, 1))
    return dog