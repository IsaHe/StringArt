import math

class NodeCircle:
    def __init__(self, center, radius, num_nodes):
        """
        center: (x0, y0) tupla de centro fijo.
        radius: radio en píxeles fijo.
        num_nodes: número de nodos (clavos).
        """
        self.cx, self.cy = center
        self.r = radius
        self.n = num_nodes

    def get_nodes(self):
        """
        Genera n puntos equiespaciados en la circunferencia.
        """
        nodes = []
        for k in range(self.n):
            theta = 2 * math.pi * k / self.n
            x = self.cx + self.r * math.cos(theta)
            y = self.cy + self.r * math.sin(theta)
            nodes.append((int(round(x)), int(round(y))))
        return nodes
