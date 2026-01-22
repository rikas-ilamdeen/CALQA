from model.image_loader import load_image

img = load_image("data/sample/leaf.jpg")
print("Image loaded successfully:", img.shape)
