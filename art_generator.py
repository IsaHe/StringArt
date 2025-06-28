import numpy as np
from typing import List, Tuple
from line_mask import LineMask

class StringArtGenerator:
    def __init__(self,
                 target: np.ndarray,
                 pins: List[Tuple[float,float]],
                 iterations: int,
                 gamma: float = 0.9):
        """
        :param target: array 2D (H×W) con valores en [0,1].
        :param pins: lista de coordenadas de pines.
        :param iterations: número de hilos a colocar.
        :param gamma: factor de oscurecimiento al trazar cada hilo.
        """
        self.target = target
        self.H, self.W = target.shape
        self.pins = pins
        self.iterations = iterations
        self.gamma = gamma
        self.canvas = np.ones_like(target)  # empieza blanco (1.0)
        self.sequence: List[Tuple[int,int]] = []

    def generate(self, start_pin: int = 0) -> List[Tuple[int,int]]:
        current = start_pin
        for _ in range(self.iterations):
            mejor_beneficio = -np.inf
            mejor_j: int = -1
            mejor_mask = None
            # probar todos los hilos desde current
            for j, pin in enumerate(self.pins):
                if j == current: continue
                rr, cc = LineMask.mask(self.pins[current], pin, self.H, self.W)
                beneficio = np.sum((self.canvas[rr,cc] - self.target[rr,cc]))
                if beneficio > mejor_beneficio:
                    mejor_beneficio = beneficio
                    mejor_j = j
                    mejor_mask = (rr, cc)
            # aplicar el hilo elegido
            rr, cc = mejor_mask  # type: ignore
            self.canvas[rr,cc] *= self.gamma
            self.sequence.append((current, mejor_j))
            current = mejor_j
        return self.sequence
