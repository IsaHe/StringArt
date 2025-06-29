import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
from datetime import datetime
import os

class StringArtGenerator:
    """
    Generador de String Art:
    - pixel_array: array 2D de valores [0,1] de la región a dibujar.
    - nodes: lista de (x,y) de posiciones de clavos (coordenadas relativas a pixel_array).
    - max_lines: número máximo de líneas a trazar.
    - output_size: tamaño (ancho=alto) del PNG de salida en píxeles.
    - use_log: si True, aplica modelo logarítmico de atenuación.
    - threshold: mejora mínima para aceptar una línea.
    """

    def __init__(self,
                 pixel_array: np.ndarray,
                 nodes: list[tuple[int,int]],
                 max_lines: int = 4000,
                 output_size: int = 1000,
                 use_log: bool = False,
                 threshold: float = 0.0):
        # Invertir escala de grises para priorizar zonas oscuras
        lum = pixel_array.copy()
        if use_log:
            self.orig = -np.log(np.clip(lum, 1e-3, 1.0))
        else:
            self.orig = 1.0 - lum

        self.coverage = np.zeros_like(self.orig)
        self.nodes = nodes
        self.max_lines = max_lines
        self.threshold = threshold
        self.h, self.w = self.orig.shape
        self.output_size = output_size

    def bresenham_line(self,
                       x0: int, y0: int,
                       x1: int, y1: int
                       ) -> list[tuple[int,int]]:
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        x, y = x0, y0
        sx, sy = (1 if x1 > x0 else -1), (1 if y1 > y0 else -1)
        line = []
        if dx > dy:
            err = dx / 2
            while x != x1:
                line.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2
            while y != y1:
                line.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        line.append((x1, y1))
        return line

    def compute_best_line(self,
                          current_node: tuple[int,int]
                          ) -> tuple[tuple[tuple[int,int], list[tuple[int,int]]], float]:
        best_pair = (None, None)
        best_improvement = 0.0
        for nx, ny in self.nodes:
            if (nx, ny) == current_node:
                continue
            line = self.bresenham_line(current_node[0], current_node[1], nx, ny)
            improvement = sum(
                max(self.orig[y, x] - self.coverage[y, x], 0.0)
                for x, y in line
            )
            if improvement > best_improvement and improvement >= self.threshold:
                best_improvement = improvement
                best_pair = ((nx, ny), line)
        return best_pair, best_improvement

    def generate(self,
                 save_path: str | None = None
                 ) -> tuple[list[tuple[tuple[int,int], list[tuple[int,int]]]], str]:
        lines = []
        current_node = self.nodes[0]

        # Bucle con tqdm hasta max_lines o hasta que no haya mejora
        for _ in tqdm(range(self.max_lines), desc="Generando string art"):
            (next_node, line), imp = self.compute_best_line(current_node)
            if next_node is None or imp <= 0.0:
                break
            for x, y in line:
                # no sobrepasar orig para coverage
                self.coverage[y, x] = min(self.coverage[y, x] + 1.0, self.orig[y, x])
            lines.append((next_node, line))
            current_node = next_node

        # Crear lienzo de salida 1000×1000 en blanco
        N = self.output_size
        output_img = Image.new('RGB', (N, N), 'white')
        draw = ImageDraw.Draw(output_img)

        # Factor de escala de coordenadas
        fx = N / self.w
        fy = N / self.h

        # Dibujar cada línea escalada
        for _, line in lines:
            scaled = [(int(x * fx), int(y * fy)) for x, y in line]
            draw.line(scaled, fill='black', width=1)

        # Guardar con timestamp si no hay ruta explícita
        if save_path is None:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            save_path = f"string_art_{ts}.png"
        else:
            base, ext = os.path.splitext(save_path)
            if ext.lower() != '.png':
                save_path = base + '.png'

        output_img.save(save_path)
        print(f"String art guardado en: {save_path}")

        return lines, save_path
