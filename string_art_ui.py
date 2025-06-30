import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from datetime import datetime
import os
import random
import cv2

from image_processor import ImageProcessor
from node_circle import NodeCircle
from string_art_generator import StringArtGenerator

class StringArtUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("String Art Generator")
        self.geometry("900x700")
        self.configure(bg='#ddd')

        # Canvas principal
        self.canvas = tk.Canvas(self, width=900, height=650, bg='white')
        self.canvas.pack(fill='both', expand=True)

        # Controles (botones y entradas)
        controls = tk.Frame(self)
        controls.place(relx=0.5, rely=0.01, anchor='n')
        self.btn_load     = tk.Button(controls, text="Cargar imagen", command=self.load_image)
        self.btn_load.pack(side='left', padx=5)

        tk.Label(controls, text="Clavos:").pack(side='left')
        self.entry_nodes  = tk.Entry(controls, width=5); self.entry_nodes.insert(0, "300")
        self.entry_nodes.pack(side='left', padx=(0,10))

        tk.Label(controls, text="Hilos:").pack(side='left')
        self.entry_lines  = tk.Entry(controls, width=5); self.entry_lines.insert(0, "4000")
        self.entry_lines.pack(side='left', padx=(0,10))

        self.btn_start    = tk.Button(controls, text="Iniciar generación", state='disabled',
                                      command=self.start_generation)
        self.btn_start.pack(side='left', padx=5)

        # Parámetros pan/zoom
        self.scale      = 1.0
        self.min_scale  = 0.2
        self.max_scale  = 5.0
        self.img_on_canvas = None
        self.img_tk        = None
        self.img_x = self.img_y = 0

        # Selector circular fijo
        self.cx, self.cy, self.cr = 450, 325, 250
        self.canvas.create_oval(
            self.cx - self.cr, self.cy - self.cr,
            self.cx + self.cr, self.cy + self.cr,
            outline='red', width=2, dash=(4,2), tag='selector_circle'
        )
        self.canvas.tag_raise('selector_circle')

        # Bindings pan/zoom sobre el Canvas
        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>",     self.do_pan)
        self.canvas.bind("<MouseWheel>",    self.do_zoom)

        # Variables de generación
        self.generator = None
        self.current_node = None
        self.lines_drawn = 0
        self.max_lines = 0

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Imágenes","*.png;*.jpg;*.jpeg;*.bmp")])
        if not path: return

        # Cargar y convertir a escala de grises
        self.processor = ImageProcessor(path)
        self.processor.to_grayscale()
        self.pil_image = self.processor.gray

        # Validar tamaño para el círculo
        w, h = self.pil_image.size
        if w <= self.cx + self.cr or h <= self.cy + self.cr:
            messagebox.showerror(
                "Error de tamaño",
                f"La imagen debe medir al menos {self.cx+self.cr+1}×{self.cy+self.cr+1} px"
            )
            return

        # Reset transform y dibujar
        self.reset_transform()
        self.draw_image()
        # Habilitar botón de inicio
        self.btn_start.config(state='normal')

    def reset_transform(self):
        self.scale = 1.0
        w, h = self.pil_image.size
        self.img_x = self.cx - w//2
        self.img_y = self.cy - h//2

    def draw_image(self):
        w, h = self.pil_image.size
        nw, nh = int(w * self.scale), int(h * self.scale)
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        img_resized = self.pil_image.resize((nw, nh), resample)
        self.img_tk = ImageTk.PhotoImage(img_resized)
        if self.img_on_canvas:
            self.canvas.delete(self.img_on_canvas)
        self.img_on_canvas = self.canvas.create_image(
            self.img_x, self.img_y, anchor='nw', image=self.img_tk
        )
        # Asegurarnos de que el círculo quede encima
        self.canvas.tag_raise('selector_circle')

    def start_pan(self, event):
        self._pan_prev = (event.x, event.y)

    def do_pan(self, event):
        px, py = self._pan_prev
        dx, dy = event.x - px, event.y - py
        self._pan_prev = (event.x, event.y)

        # Nueva posición con clamping
        new_x = self.img_x + dx
        new_y = self.img_y + dy
        w, h = self.img_tk.width(), self.img_tk.height()
        min_x = self.cx - (w - self.cr)
        max_x = self.cx + self.cr
        min_y = self.cy - (h - self.cr)
        max_y = self.cy + self.cr
        self.img_x = min(max(new_x, min_x), max_x)
        self.img_y = min(max(new_y, min_y), max_y)

        self.canvas.coords(self.img_on_canvas, self.img_x, self.img_y)

    def do_zoom(self, event):
        factor = 1.1 if event.delta > 0 else 0.9
        new_scale = self.scale * factor
        if not (self.min_scale <= new_scale <= self.max_scale):
            return
        cx, cy = event.x, event.y
        ix, iy = self.canvas.coords(self.img_on_canvas)
        self.scale = new_scale
        self.img_x = cx - (cx - ix) * factor
        self.img_y = cy - (cy - iy) * factor
        self.draw_image()

    def start_generation(self):
        # Recorte de la región circular
        cx_img = int((self.cx - self.img_x) / self.scale)
        cy_img = int((self.cy - self.img_y) / self.scale)
        r_img  = int(self.cr / self.scale)
        box = (
            cx_img - r_img, cy_img - r_img,
            cx_img + r_img, cy_img + r_img
        )
        crop_gray = self.pil_image.crop(box)

        # Parámetros de nodos y líneas
        num_nodes = int(self.entry_nodes.get())
        self.max_lines = int(self.entry_lines.get())
        w_crop, h_crop = crop_gray.size
        radius = min(w_crop, h_crop)//2 - 1
        center = (w_crop//2, h_crop//2)
        nodes = NodeCircle(center, radius, num_nodes).get_nodes()

        # Preprocesar el array de grises
        arr = np.array(crop_gray).astype(float)/255.0

        # Crear generador
        self.generator    = StringArtGenerator(arr, nodes,
                                               max_lines=self.max_lines,
                                               output_size=1000,
                                               use_log=False)
        self.current_node = random.choice(nodes)
        self.lines_drawn   = 0

        # Limpiar cualquier línea previa y desactivar botón
        self.canvas.delete('line_segment')
        self.btn_start.config(state='disabled')

        # Calcular factores de escalado para previsualización (600×600)
        self.fx = 600 / w_crop
        self.fy = 600 / h_crop

        # Dibujar la mini-previsualización en el Canvas
        self.preview_tk = ImageTk.PhotoImage(crop_gray.resize((600,600)))
        # Borra la anterior si existe
        self.canvas.delete('preview')
        self.canvas.create_image(150, 25, anchor='nw', image=self.preview_tk, tags='preview')

        # Lanzar la animación
        self.after(1, self.step_generation)

    def step_generation(self):
        # Una iteración: escoger línea
        (next_node, line), imp = self.generator.compute_best_line(self.current_node)
        if next_node is None:
            # Fallback aleatorio
            nodes = self.generator.nodes
            next_node = random.choice([n for n in nodes if n!=self.current_node])
            line = self.generator.bresenham_line(self.current_node[0],
                                                 self.current_node[1],
                                                 next_node[0],
                                                 next_node[1])
        # Actualizar coverage
        self.generator.update_coverage(line)

        # Dibujar segmento recto en preview
        x0,y0 = line[0]; x1,y1 = line[-1]
        X0 = int(x0*self.fx)+150; Y0 = int(y0*self.fy)+25
        X1 = int(x1*self.fx)+150; Y1 = int(y1*self.fy)+25
        self.canvas.create_line(X0, Y0, X1, Y1,
                                fill='black', width=1,
                                tags='line_segment')

        self.current_node = next_node
        self.lines_drawn += 1

        if self.lines_drawn < self.max_lines:
            # Programar siguiente paso
            self.after(1, self.step_generation)
        else:
            # Guardar final
            _, save_path = self.generator.generate(save_path=None)
            messagebox.showinfo("¡Listo!",
                                f"String art completo guardado en:\n{save_path}")

if __name__ == "__main__":
    app = StringArtUI()
    app.mainloop()
