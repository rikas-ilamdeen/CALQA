from skimage import io

def load_image(image_path):
    """
    Load an image from disk
    """
    image = io.imread(image_path)
    if image is None:
        raise FileNotFoundError("Image not found")
    return image
