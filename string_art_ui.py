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

        # Canvas y controles
        self.canvas = tk.Canvas(self, width=900, height=650, bg='white')
        self.canvas.pack(fill='both', expand=True)

        ctrl = tk.Frame(self)
        ctrl.place(relx=0.5, rely=0.01, anchor='n')
        tk.Button(ctrl, text="Cargar imagen", command=self.load_image).pack(side='left', padx=5)
        tk.Label(ctrl, text="Clavos:").pack(side='left')
        self.entry_nodes = tk.Entry(ctrl, width=5); self.entry_nodes.insert(0, "300")
        self.entry_nodes.pack(side='left', padx=(0,10))
        tk.Label(ctrl, text="Hilos:").pack(side='left')
        self.entry_lines = tk.Entry(ctrl, width=5); self.entry_lines.insert(0, "4000")
        self.entry_lines.pack(side='left', padx=(0,10))
        self.btn_start = tk.Button(ctrl, text="Iniciar generación", state='disabled',
                                   command=self.start_generation)
        self.btn_start.pack(side='left', padx=5)

        # Pan/zoom
        self.scale = 1.0; self.min_scale = 0.2; self.max_scale = 5.0
        self.img_on_canvas = None; self.img_tk = None
        self.img_x = self.img_y = 0

        # Selector circular fijo
        self.cx, self.cy, self.cr = 450, 325, 250
        self.canvas.create_oval(
            self.cx-self.cr, self.cy-self.cr,
            self.cx+self.cr, self.cy+self.cr,
            outline='red', width=2, dash=(4,2), tag='selector_circle'
        )
        self.canvas.tag_raise('selector_circle')

        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>",     self.do_pan)
        self.canvas.bind("<MouseWheel>",    self.do_zoom)

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Imágenes","*.png;*.jpg;*.jpeg;*.bmp")]
        )
        if not path: return

        self.processor = ImageProcessor(path)
        self.processor.to_grayscale()
        self.pil_image = self.processor.gray
        w, h = self.pil_image.size
        if w <= self.cx+self.cr or h <= self.cy+self.cr:
            messagebox.showerror("Error de tamaño",
                f"La imagen debe ser mayor que {self.cx+self.cr+1}×{self.cy+self.cr+1} px")
            return

        self.reset_transform(); self.draw_image()
        self.btn_start.config(state='normal')

    def reset_transform(self):
        w, h = self.pil_image.size
        self.scale = 1.0
        self.img_x = self.cx - w//2
        self.img_y = self.cy - h//2

    def draw_image(self):
        w, h = self.pil_image.size
        nw, nh = int(w*self.scale), int(h*self.scale)
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
        self.canvas.tag_raise('selector_circle')

    def start_pan(self, e):
        self._pan_prev = (e.x, e.y)

    def do_pan(self, e):
        px, py = self._pan_prev; dx, dy = e.x-px, e.y-py
        self._pan_prev = (e.x, e.y)
        new_x, new_y = self.img_x+dx, self.img_y+dy
        w, h = self.img_tk.width(), self.img_tk.height()
        min_x, max_x = self.cx-(w-self.cr), self.cx+self.cr
        min_y, max_y = self.cy-(h-self.cr), self.cy+self.cr
        self.img_x = min(max(new_x, min_x), max_x)
        self.img_y = min(max(new_y, min_y), max_y)
        self.canvas.coords(self.img_on_canvas, self.img_x, self.img_y)

    def do_zoom(self, e):
        factor = 1.1 if e.delta>0 else 0.9
        new_s = self.scale*factor
        if not (self.min_scale<=new_s<=self.max_scale): return
        cx, cy = e.x, e.y
        ix, iy = self.canvas.coords(self.img_on_canvas)
        self.scale = new_s
        self.img_x = cx - (cx-ix)*factor
        self.img_y = cy - (cy-iy)*factor
        self.draw_image()

    def start_generation(self):
        # --- Preprocesado avanzado ---
        cx_i = int((self.cx - self.img_x)/self.scale)
        cy_i = int((self.cy - self.img_y)/self.scale)
        r_i  = int(self.cr / self.scale)
        box = (cx_i-r_i, cy_i-r_i, cx_i+r_i, cy_i+r_i)
        crop_gray = self.pil_image.crop(box)

        gray = np.array(crop_gray).astype(np.uint8)
        hist_eq = cv2.equalizeHist(gray)                                 # :contentReference[oaicite:20]{index=20}
        clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        clahe_eq= clahe.apply(hist_eq)                                    # :contentReference[oaicite:21]{index=21}
        blur    = cv2.GaussianBlur(clahe_eq, (5,5), sigmaX=0)             # :contentReference[oaicite:22]{index=22}
        edges   = cv2.Canny(blur, 50, 150)/255.0                          # :contentReference[oaicite:23]{index=23}

        norm_blur = blur.astype(float)/255.0                              # :contentReference[oaicite:24]{index=24}
        weight    = 0.7*norm_blur + 0.3*edges                             # 

        h_w, w_w = weight.shape
        mask = np.ones_like(weight)*0.5                                   # 
        mask[h_w//2:,:] = weight[h_w//2:,:]
        arr = mask

        # Parámetros de nodos y generator
        num_nodes = int(self.entry_nodes.get())
        self.max_lines = int(self.entry_lines.get())
        radius = min(h_w, w_w)//2 - 1
        center = (w_w//2, h_w//2)
        nodes  = NodeCircle(center, radius, num_nodes).get_nodes()

        self.generator = StringArtGenerator(arr, nodes,
                                            max_lines=self.max_lines,
                                            output_size=1000,
                                            use_log=False)
        self.current_node = random.choice(nodes)
        self.lines_drawn = 0

        # Preview blanco 600×600
        self.canvas.delete('preview_bg','line_segment')
        self.canvas.create_rectangle(
            150,25,150+600,25+600,
            fill='white', outline='', tags='preview_bg'
        )
        self.fx = 600 / w_w; self.fy = 600 / h_w

        self.btn_start.config(state='disabled')
        self.after(1, self.step_generation)   # inicia animación 

    def step_generation(self):
        (n_node, line), imp = self.generator.compute_best_line(self.current_node)
        if n_node is None:
            n_node = random.choice([n for n in self.generator.nodes if n!=self.current_node])
            line = self.generator.bresenham_line(self.current_node[0],
                                                 self.current_node[1],
                                                 n_node[0], n_node[1])
        self.generator.update_coverage(line)

        x0,y0 = line[0]; x1,y1 = line[-1]
        X0 = int(x0*self.fx)+150; Y0 = int(y0*self.fy)+25
        X1 = int(x1*self.fx)+150; Y1 = int(y1*self.fy)+25
        self.canvas.create_line(X0,Y0,X1,Y1,
                                fill='black', width=1,
                                tags='line_segment')                # :contentReference[oaicite:28]{index=28}

        self.current_node = n_node
        self.lines_drawn += 1

        if self.lines_drawn < self.max_lines:
            self.after(1, self.step_generation)
        else:
            # Guardado final en alta resolución con tqdm
            _, save_path = self.generator.generate(save_path=None)
            messagebox.showinfo("¡Listo!",
                                f"String art final guardado en:\n{save_path}")

if __name__ == "__main__":
    app = StringArtUI()
    app.mainloop()
