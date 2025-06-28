import numpy as np
from skimage.draw import line
from typing import Tuple

class LineMask:
    @staticmethod
    def mask(pin1: Tuple[float,float],
             pin2: Tuple[float,float],
             height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Devuelve dos arrays (rr, cc) con las filas y columnas
        de los píxeles que cruza la línea pin1→pin2.
        """
        x0, y0 = pin1; x1, y1 = pin2
        rr, cc = line(int(y0), int(x0), int(y1), int(x1))
        rr = np.clip(rr, 0, height - 1)
        cc = np.clip(cc, 0, width  - 1)
        return rr, cc
