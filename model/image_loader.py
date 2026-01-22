import cv2

def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError("Image not found")
    return image
