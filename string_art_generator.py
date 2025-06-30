import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image, ImageDraw
from datetime import datetime
import os
import random

class StringArtGenerator:
    """
    Genera hiloramas:
    - pixel_array: 2D array de grises [0,1].
    - nodes: lista de (x,y) de clavos en coords de pixel_array.
    - max_lines: número máximo de hilos.
    - output_size: resolución final (px).
    - use_log: si True, aplica -log(luminosidad).
    """

    def __init__(self,
                 pixel_array: np.ndarray,
                 nodes: list[tuple[int,int]],
                 max_lines: int = 4000,
                 output_size: int = 1000,
                 use_log: bool = False):
        lum = pixel_array.copy()
        # 1) Crear mapa de importancia: mezcla inversión de grises y bordes Canny :contentReference[oaicite:6]{index=6}
        edges = cv2.Canny((lum*255).astype(np.uint8), 50, 150)/255.0  # cv2.Canny :contentReference[oaicite:7]{index=7}
        inv = (-np.log(np.clip(lum,1e-3,1.0)) if use_log else 1.0 - lum)  # inv. grises :contentReference[oaicite:8]{index=8}
        self.weight = 0.7*inv + 0.3*edges

        # 2) Cobertura y parámetros
        self.coverage = np.zeros_like(self.weight)
        self.nodes = nodes
        self.max_lines = max_lines
        self.h, self.w = self.weight.shape
        self.output_size = output_size

    def bresenham_line(self, x0, y0, x1, y1):
        """
        Algoritmo de Bresenham para rasterizar líneas (sólo para coverage) :contentReference[oaicite:9]{index=9}.
        """
        dx, dy = abs(x1-x0), abs(y1-y0)
        x, y = x0, y0
        sx, sy = (1 if x1>x0 else -1), (1 if y1>y0 else -1)
        line = []
        if dx>dy:
            err = dx/2
            while x!=x1:
                line.append((x,y))
                err -= dy
                if err<0:
                    y+=sy; err+=dx
                x+=sx
        else:
            err = dy/2
            while y!=y1:
                line.append((x,y))
                err -= dx
                if err<0:
                    x+=sx; err+=dy
                y+=sy
        line.append((x1,y1))
        return line

    def compute_best_line(self, current_node):
        """
        Elige la línea con mayor suma de max(weight-coverage,0). :contentReference[oaicite:10]{index=10}
        """
        best_pair, best_imp = (None,None), -1.0
        for nx,ny in self.nodes:
            if (nx,ny)==current_node: continue
            line = self.bresenham_line(current_node[0], current_node[1], nx, ny)
            imp = sum(max(self.weight[y,x]-self.coverage[y,x], 0.0) for x,y in line)
            if imp>best_imp:
                best_imp, best_pair = imp, ((nx,ny), line)
        return best_pair, best_imp

    def update_coverage(self, line):
        """
        Actualiza coverage sin exceder weight :contentReference[oaicite:11]{index=11}.
        """
        for x,y in line:
            self.coverage[y,x] = min(self.coverage[y,x]+1, self.weight[y,x])

    def generate(self, save_path=None):
        """
        Genera exactamente max_lines iteraciones (sin break) con tqdm :contentReference[oaicite:12]{index=12}.
        Devuelve lista de segmentos (endpoints) y ruta de PNG salvado.
        """
        segments = []
        current = random.choice(self.nodes)

        for _ in tqdm(range(self.max_lines), desc="Generando string art"):
            (next_node,line), imp = self.compute_best_line(current)
            if next_node is None:
                # Fallback aleatorio si no hay mejora
                next_node = random.choice([n for n in self.nodes if n!=current])
                line = self.bresenham_line(current[0],current[1], next_node[0], next_node[1])
            self.update_coverage(line)
            segments.append((current,next_node))
            current = next_node

        # 3) Dibujo final en lienzo 1000×1000 px
        N = self.output_size
        canvas = Image.new('RGB',(N,N),'white')
        draw = ImageDraw.Draw(canvas)
        fx, fy = N/self.w, N/self.h

        for (x0,y0),(x1,y1) in segments:
            X0,Y0 = int(x0*fx), int(y0*fy)
            X1,Y1 = int(x1*fx), int(y1*fy)
            draw.line((X0,Y0,X1,Y1), fill='black', width=1)  # líneas rectas :contentReference[oaicite:13]{index=13}

        # 4) Guardado PNG con timestamp
        if save_path is None:
            save_path = f"string_art_{datetime.now():%Y%m%d-%H%M%S}.png"  # timestamp :contentReference[oaicite:14]{index=14}
        else:
            base,ext = os.path.splitext(save_path)
            if ext.lower()!='.png': save_path = base+'.png'

        canvas.save(save_path)
        print(f"String art guardado en: {save_path}")
        return segments, save_path
