from PIL import Image
import numpy as np

class ImagePreprocessor:
    def __init__(self, target_size: int):
        """
        :param target_size: Tamaño máximo (px) al que escalar la imagen.
        """
        self.target_size = target_size

    def load_and_resize(self, path: str) -> Image:
        img = Image.open(path).convert('RGB')
        w, h = img.size
        scale = self.target_size / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        return img.resize(new_size, Image.LANCZOS)

    def to_gray_array(self, img: Image.Image) -> np.ndarray:
        """
        Convierte la imagen PIL a un array float32 en escala de grises [0,1].
        """
        gray = img.convert('L')
        arr = np.asarray(gray, dtype=np.float32) / 255.0
        return arr
