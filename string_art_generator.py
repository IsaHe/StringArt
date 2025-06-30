import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image, ImageDraw
from datetime import datetime
import os
import random

class StringArtGenerator:
    """
    Generador de String Art:
    - pixel_array: array 2D de grises normalizados [0,1].
    - nodes: posiciones de clavos (x,y) en coords de pixel_array.
    - max_lines: número máximo de hilos a trazar.
    - output_size: tamaño final de la imagen PNG.
    - use_log: aplica -log(lum) en lugar de inv. lineal de grises.
    """

    def __init__(self,
                 pixel_array: np.ndarray,
                 nodes: list[tuple[int,int]],
                 max_lines: int = 4000,
                 output_size: int = 1000,
                 use_log: bool = False):
        lum = pixel_array.copy()
        # Mapa de importancia: mezcla inv. de grises y bordes Canny
        edges = cv2.Canny((lum*255).astype(np.uint8), 50, 150) / 255.0  # :contentReference[oaicite:13]{index=13}
        inv = (-np.log(np.clip(lum,1e-3,1.0)) if use_log else 1.0 - lum)  # :contentReference[oaicite:14]{index=14}
        self.weight = 0.7 * inv + 0.3 * edges                            # 

        # Cobertura inicial y otros parámetros
        self.coverage = np.zeros_like(self.weight)
        self.nodes    = nodes
        self.max_lines= max_lines
        self.h, self.w = self.weight.shape
        self.output_size = output_size

    def bresenham_line(self, x0, y0, x1, y1):
        """
        Bresenham para calcular cobertura por píxel (sin dibujar).
        """
        dx, dy = abs(x1-x0), abs(y1-y0)
        x, y = x0, y0
        sx, sy = (1 if x1>x0 else -1), (1 if y1>y0 else -1)
        line = []
        if dx > dy:
            err = dx/2
            while x != x1:
                line.append((x,y))
                err -= dy
                if err < 0:
                    y += sy; err += dx
                x += sx
        else:
            err = dy/2
            while y != y1:
                line.append((x,y))
                err -= dx
                if err < 0:
                    x += sx; err += dy
                y += sy
        line.append((x1,y1))
        return line

    def compute_best_line(self, current_node):
        """
        Selecciona la línea cuyo segmento aporta la mayor sumatoria de
        max(weight−coverage,0) (solo contribuciones positivas). 
        """
        best_pair, best_imp = (None, None), -1.0
        for nx, ny in self.nodes:
            if (nx, ny) == current_node:
                continue
            line = self.bresenham_line(current_node[0], current_node[1], nx, ny)
            imp = sum(max(self.weight[y, x] - self.coverage[y, x], 0.0)
                      for x, y in line)
            if imp > best_imp:
                best_imp = imp
                best_pair = ((nx, ny), line)
        return best_pair, best_imp

    def update_coverage(self, line):
        """
        Incrementa coverage en cada píxel hasta no sobrepasar weight. 
        """
        for x, y in line:
            self.coverage[y, x] = min(self.coverage[y, x] + 1.0, self.weight[y, x])

    def generate(self, save_path=None):
        """
        Traza exactamente max_lines segmentos (sin break) mostrando tqdm
        y luego dibuja líneas rectas en PIL 1000×1000 px y guarda PNG. 
        """
        segments = []
        current = random.choice(self.nodes)

        for _ in tqdm(range(self.max_lines), desc="Generando string art"):
            (next_node, line), imp = self.compute_best_line(current)
            if next_node is None:
                next_node = random.choice([n for n in self.nodes if n!=current])
                line = self.bresenham_line(current[0], current[1],
                                          next_node[0], next_node[1])
            self.update_coverage(line)
            segments.append((current, next_node))
            current = next_node

        # Dibujo final en alta resolución
        N = self.output_size
        canvas = Image.new('RGB', (N, N), 'white')
        draw = ImageDraw.Draw(canvas)
        fx, fy = N / self.w, N / self.h

        for (x0, y0), (x1, y1) in segments:
            X0, Y0 = int(x0*fx), int(y0*fy)
            X1, Y1 = int(x1*fx), int(y1*fy)
            draw.line((X0, Y0, X1, Y1), fill='black', width=1)        # :contentReference[oaicite:19]{index=19}

        # Guardar con timestamp automático
        if save_path is None:
            save_path = f"string_art_{datetime.now():%Y%m%d-%H%M%S}.png"
        else:
            base, ext = os.path.splitext(save_path)
            if ext.lower() != '.png':
                save_path = base + '.png'

        canvas.save(save_path)
        print(f"String art guardado en: {save_path}")
        return segments, save_path
