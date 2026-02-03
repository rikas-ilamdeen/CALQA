import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

class MainWindow:
    """
    View layer: Tkinter-based GUI for user interaction
    Allows filter selection and displays results
    """
    def __init__(self, root, controller):
        self.controller = controller
        self.root = root
        self.root.title("CALQA – Centella Asiatica Leaves Quality Assessment")
        self.root.geometry("900x600")

        tk.Button(root, text="Load Image", command=self.load_image).pack(pady=5)

        # Radio buttons instead of checkboxes
        self.method = tk.StringVar(value="LoG")

        tk.Label(root, text="Select Filter").pack()
        tk.Radiobutton(root, text="Laplacian of Gaussian (LoG)", 
                       variable=self.method, value="LoG").pack()
        tk.Radiobutton(root, text="Difference of Gaussian (DoG)", 
                       variable=self.method, value="DoG").pack()

        tk.Button(root, text="Compute", command=self.compute).pack(pady=10)

        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.time_label = tk.Label(root, text="")
        self.time_label.pack(pady=5)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png")])
        if path:
            self.controller.load_image(path)

    def show_image(self, image):
        self.display_image(image)

    def compute(self):
        self.controller.process(self.method.get())

    def show_result(self, image):
        self.display_image(image)

    def show_time(self, time_ms):
        self.time_label.config(text=f"Total Processing Time: {time_ms:.2f} ms")

    def display_image(self, image):
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        if image.ndim == 2:
            img = Image.fromarray(image, mode="L")
        else:
            img = Image.fromarray(image)

        img = img.resize((400, 300))
        img_tk = ImageTk.PhotoImage(img)

        self.image_label.img_tk = img_tk
        self.image_label.config(image=img_tk)
