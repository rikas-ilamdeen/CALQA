from model.image_loader import load_image
from model.preprocessing import to_grayscale
from model.log_cpu import apply_log
from model.dog_cpu import apply_dog
from model.timer import Timer

class AppController:
    """
    Controller layer: coordinates data flow between View and Model
    """
    def __init__(self, view):
        self.view = view
        self.image_path = None
        self.image = None

    def load_image(self, path):
        self.image_path = path
        self.image = load_image(path)

        # Show original image
        self.view.show_result(self.image)

    def process(self, method):
        # Measure time
        with Timer() as t:

            # Reload image 
            image = load_image(self.image_path)

            # Convert to grayscale (required for edge detection)
            gray = to_grayscale(image)

            # Select CPU-based filter
            if method == "LoG":
                output = apply_log(gray)
            elif method == "DoG":
                output = apply_dog(gray) 
            else:
                output = gray
            
            # Display result
            self.view.show_result(output)

        # Show total elapsed time
        self.view.show_time(t.elapsed_ms)
