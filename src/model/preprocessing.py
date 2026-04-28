from skimage.color import rgb2gray
from skimage.filters import gaussian

def to_grayscale(image):
    # Converts RGB → grayscale in range [0,1]
    return rgb2gray(image)

# normalization
def gaussian_blur(gray, sigma=1.0):
    return gaussian(gray, sigma=sigma)
