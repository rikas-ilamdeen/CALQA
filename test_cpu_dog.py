from model.image_loader import load_image
from model.dog_cpu import dog_cpu
import matplotlib.pyplot as plt

IMAGE_PATH = "data/samples/leaf.jpg"

image = load_image(IMAGE_PATH)
dog_image, time_ms = dog_cpu(image)

print(f"CPU DoG Processing Time: {time_ms:.2f} ms")

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("CPU DoG Result")
plt.imshow(dog_image, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()
