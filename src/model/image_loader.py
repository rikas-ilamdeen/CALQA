from skimage import io

# Load image from disk
def load_image(path):
    image = io.imread(path)
    if image is None:
        raise ValueError("Failed to load image")
    return image
