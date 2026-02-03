from skimage import io

def load_image(path):
    image = io.imread(path)
    if image is None:
        raise ValueError("Failed to load image")
    return image
