import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import os
from datetime import datetime

from image_processor import ImageProcessor
from node_circle import NodeCircle
from string_art_generator import StringArtGenerator

class StringArtUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("String Art Generator")
        self.geometry("900x650")
        self.configure(bg='#ddd')

        # Canvas principal
        self.canvas = tk.Canvas(self, width=900, height=650, bg='white')
        self.canvas.pack(fill='both', expand=True)

        # Frame de controles (botones + entradas)
        controls = tk.Frame(self)
        controls.place(relx=0.5, rely=0.01, anchor='n')

        # Botón para cargar imagen
        self.btn_load     = tk.Button(controls, text="Cargar imagen", command=self.load_image)
        self.btn_load.pack(side='left', padx=5)

        # Entrada para número de clavos
        tk.Label(controls, text="Clavos:").pack(side='left')
        self.entry_nodes  = tk.Entry(controls, width=5)
        self.entry_nodes.insert(0, "300")
        self.entry_nodes.pack(side='left', padx=(0,10))
        # get() recupera el texto del Entry :contentReference[oaicite:4]{index=4}

        # Entrada para número de hilos
        tk.Label(controls, text="Hilos:").pack(side='left')
        self.entry_lines  = tk.Entry(controls, width=5)
        self.entry_lines.insert(0, "4000")
        self.entry_lines.pack(side='left', padx=(0,10))
        # get() recupera el texto del Entry :contentReference[oaicite:5]{index=5}

        # Botón para generar arte
        self.btn_generate = tk.Button(controls, text="Generar arte", state='disabled', command=self.generate_art)
        self.btn_generate.pack(side='left', padx=5)

        # Parámetros de transformación de la imagen
        self.scale         = 1.0
        self.min_scale     = 0.2
        self.max_scale     = 5.0
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

        # Bindings para pan manual (solo imagen) y zoom
        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>",     self.do_pan)
        self.canvas.bind("<MouseWheel>",    self.do_zoom)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Imágenes","*.png;*.jpg;*.jpeg;*.bmp")])
        if not path:
            return
        self.processor = ImageProcessor(path)
        self.processor.to_grayscale()
        self.pil_image = self.processor.gray  # imagen L mode

        # Validar dimensiones suficientes para el círculo
        w, h = self.pil_image.size
        if w <= self.cx + self.cr or h <= self.cy + self.cr:
            messagebox.showerror(
                "Error de tamaño",
                f"La imagen debe ser mayor que {self.cx+self.cr}x{self.cy+self.cr} px (700x575)."
            )
            return
        # Habilitar generación
        self.reset_transform()
        self.draw_image()
        self.btn_generate.config(state='normal')

    def reset_transform(self):
        self.scale = 1.0
        w, h = self.pil_image.size
        self.img_x = self.cx - w//2
        self.img_y = self.cy - h//2

    def draw_image(self):
        w, h = self.pil_image.size
        nw, nh = int(w*self.scale), int(h*self.scale)
        # Reemplaza ANTIALIAS por LANCZOS tras Pillow 10 :contentReference[oaicite:6]{index=6}
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        img_resized = self.pil_image.resize((nw, nh), resample)
        self.img_tk = ImageTk.PhotoImage(img_resized)
        if self.img_on_canvas:
            self.canvas.delete(self.img_on_canvas)
        self.img_on_canvas = self.canvas.create_image(self.img_x, self.img_y, anchor='nw', image=self.img_tk)
        self.canvas.tag_raise('selector_circle')

    def start_pan(self, event):
        self._pan_prev = (event.x, event.y)

    def do_pan(self, event):
        prev_x, prev_y = self._pan_prev
        dx, dy = event.x - prev_x, event.y - prev_y
        self._pan_prev = (event.x, event.y)

        new_x = self.img_x + dx
        new_y = self.img_y + dy
        w, h = self.img_tk.width(), self.img_tk.height()

        # Clampeo para que el círculo siempre quede dentro de la imagen :contentReference[oaicite:7]{index=7}
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
        if self.min_scale <= new_scale <= self.max_scale:
            cx, cy = event.x, event.y
            ix, iy = self.canvas.coords(self.img_on_canvas)
            self.scale = new_scale
            self.img_x = cx - (cx - ix) * factor
            self.img_y = cy - (cy - iy) * factor
            self.draw_image()

    def generate_art(self):
        # 1) Recorte y guardado de la imagen de referencia en gris
        cx_img = int((self.cx - self.img_x) / self.scale)
        cy_img = int((self.cy - self.img_y) / self.scale)
        r_img  = int(self.cr / self.scale)
        box = (
            cx_img - r_img, cy_img - r_img,
            cx_img + r_img, cy_img + r_img
        )
        crop_gray = self.pil_image.crop(box)  # PIL.Image.crop :contentReference[oaicite:6]{index=6}
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")      # Timestamp 
        base = os.path.splitext(self.processor.path)[0]    # splitext → (root, ext) :contentReference[oaicite:8]{index=8}
        ref_path = f"{base}_ref_gray_{ts}.png"
        crop_gray.save(ref_path)                           # Guardar referencia :contentReference[oaicite:9]{index=9}
        print(f"Referencia B/N guardada en: {ref_path}")

        # 2) Convertir el recorte a array NumPy normalizado
        arr_crop = np.array(crop_gray).astype(float) / 255.0  # PIL→NumPy :contentReference[oaicite:10]{index=10}

        # 3) Parámetros de clavos y hilos
        try:
            num_nodes = int(self.entry_nodes.get())
            max_lines = int(self.entry_lines.get())
        except ValueError:
            messagebox.showerror("Parámetro inválido", "Clavos e Hilos deben ser enteros.")
            return                                             # Entry.get() → string :contentReference[oaicite:11]{index=11}

        # 4) Recalcular center/radius desde el recorte para nodos válidos
        w_crop, h_crop = crop_gray.size
        radius = min(w_crop, h_crop) // 2 - 1                # Evitar índice = dimension :contentReference[oaicite:12]{index=12}
        center = (w_crop // 2, h_crop // 2)
        nodes = NodeCircle(center, radius, num_nodes).get_nodes()

        # 5) Generar y autoguardar string art
        generator = StringArtGenerator(arr_crop, nodes, max_lines)
        lines, art_path = generator.generate(save_path=None)  # Guarda con timestamp :contentReference[oaicite:13]{index=13}

        # 6) Mostrar resultado alineado con el selector circular
        result_img = Image.open(art_path)
        result_tk  = ImageTk.PhotoImage(result_img)
        self.canvas.create_image(
            self.cx - self.cr, self.cy - self.cr,
            anchor='nw', image=result_tk, tags='result_string_art'
        )
        self.canvas.image = result_tk
        print(f"String art guardado en: {art_path}")


if __name__ == "__main__":
    app = StringArtUI()
    app.mainloop()
