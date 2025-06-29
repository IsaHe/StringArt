import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
from datetime import datetime
import os

class StringArtGenerator:
    """
    Generador de String Art:
    - pixel_array: array 2D de valores [0,1] de la región a dibujar.
    - nodes: lista de (x,y) de posiciones de clavos.
    - max_lines: número máximo de líneas a trazar.
    - use_log: si True, usa modelo logarítmico de atenuación de luz.
    - threshold: mejora mínima para aceptar una línea.
    """

    def __init__(self,
                 pixel_array: np.ndarray,
                 nodes: list[tuple[int,int]],
                 max_lines: int = 4000,
                 use_log: bool = False,
                 threshold: float = 0.0):
        # Invertir la escala de grises: 1 = negro, 0 = blanco :contentReference[oaicite:0]{index=0}
        lum = pixel_array.copy()
        if use_log:
            # Modelo logarítmico: -log(luminosity) :contentReference[oaicite:1]{index=1}
            self.orig = -np.log(np.clip(lum, 1e-3, 1.0))
        else:
            self.orig = 1.0 - lum

        # Matriz de cobertura inicial (número de hilos que pasan por cada píxel)
        self.coverage = np.zeros_like(self.orig)
        self.nodes = nodes
        self.max_lines = max_lines
        self.threshold = threshold
        self.h, self.w = self.orig.shape

    def bresenham_line(self,
                       x0: int, y0: int,
                       x1: int, y1: int
                       ) -> list[tuple[int,int]]:
        """
        Algoritmo de Bresenham para obtener todos los píxeles entre dos puntos :contentReference[oaicite:2]{index=2}.
        """
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
        """
        Para cada posible hilo desde current_node calcula la mejora en 'oscuridad'
        y devuelve la mejor (next_node, línea) junto con su valor de mejora :contentReference[oaicite:3]{index=3}.
        """
        best_pair = (None, None)
        best_improvement = 0.0
        for nx, ny in self.nodes:
            if (nx, ny) == current_node:
                continue
            line = self.bresenham_line(current_node[0], current_node[1], nx, ny)
            # Mejora = suma de orig - coverage a lo largo de la línea
            improvement = sum(self.orig[y, x] - self.coverage[y, x] for x, y in line)
            if improvement > best_improvement and improvement >= self.threshold:
                best_improvement = improvement
                best_pair = ((nx, ny), line)
        return best_pair, best_improvement

    def generate(self,
                 save_path: str | None = None
                 ) -> tuple[list[tuple[tuple[int,int], list[tuple[int,int]]]], str]:
        """
        Genera pasadas de hilo (greedy) con barra de progreso y:
        1) Dibuja el resultado en un PIL.Image.
        2) Guarda automáticamente como PNG.
        Devuelve (líneas, ruta_guardado) .
        """
        lines = []
        current_node = self.nodes[0]

        # Iterar siempre max_lines veces (o menos si no hay mejora significativa)
        for _ in tqdm(range(self.max_lines), desc="Generando string art"):
            (next_node, line), imp = self.compute_best_line(current_node)
            if next_node is None:
                break
            # Actualizar cobertura (conteo de cuántos hilos pasan)
            for x, y in line:
                self.coverage[y, x] += 1
            lines.append((next_node, line))
            current_node = next_node

        # Crear lienzo en blanco para salida final
        output_img = Image.new('RGB', (self.w, self.h), 'white')
        draw = ImageDraw.Draw(output_img)

        # Dibujar cada línea trazada
        for _, line in lines:
            draw.line(line, fill='black', width=1)

        # Determinar ruta de guardado con timestamp
        if save_path is None:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")  # formateo de fecha 
            save_path = f"string_art_{ts}.png"
        else:
            base, ext = os.path.splitext(save_path)
            if ext.lower() != '.png':
                save_path = base + '.png'

        # Guardar PNG resultante
        output_img.save(save_path)
        print(f"String art guardado en: {save_path}")

        return lines, save_path
