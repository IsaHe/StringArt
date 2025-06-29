import numpy as np
from PIL import Image

class ImageProcessor:
    def __init__(self, path):
        """Carga la imagen desde disco."""
        self.image = Image.open(path).convert('RGB')
        self.path = path 

    def to_grayscale(self):
        """Convierte la imagen a escala de grises."""
        self.gray = self.image.convert('L')
        return self.gray

    def get_array(self):
        """Devuelve la imagen en escala de grises como array NumPy con valores entre 0 y 1."""
        arr = np.array(self.gray, dtype=np.float32)
        return arr / 255.0

