import math
from typing import List, Tuple

class PinManager:
    def __init__(self, center: Tuple[float,float], radius: float, num_pins: int):
        """
        :param center: (x,y) del centro del círculo.
        :param radius: radio del círculo (en px).
        :param num_pins: número de pines (nodos) en la circunferencia.
        """
        self.cx, self.cy = center
        self.radius = radius
        self.num_pins = num_pins
        self.pins = self._calculate_pins()

    def _calculate_pins(self) -> List[Tuple[float,float]]:
        pins = []
        for i in range(self.num_pins):
            θ = 2 * math.pi * i / self.num_pins
            x = self.cx + self.radius * math.cos(θ)
            y = self.cy + self.radius * math.sin(θ)
            pins.append((x, y))
        return pins

    def get_pins(self) -> List[Tuple[float,float]]:
        return self.pins
