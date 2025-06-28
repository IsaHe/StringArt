from PIL import Image, ImageDraw
from typing import List, Tuple

class PreviewRenderer:
    def __init__(self, size: int):
        """
        :param size: ancho y alto del lienzo (px).
        """
        self.size = size

    def render(self,
               pins: List[Tuple[float,float]],
               sequence: List[Tuple[int,int]],
               line_width: int = 1) -> Image.Image:
        img = Image.new('RGB', (self.size, self.size), 'white')
        draw = ImageDraw.Draw(img)
        for i, j in sequence:
            draw.line([pins[i], pins[j]], fill='black', width=line_width)
        return img
