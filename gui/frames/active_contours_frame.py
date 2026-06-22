"""Active contours (pixel exchange) segmentation frame."""
import tkinter as tk
import customtkinter as ctk
import numpy as np
import cv2
import threading
from tkinter import filedialog
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from gui.utils import convert_cv_to_ctk
from processing.active_contours import segmentar_imagen, estimar_parametros


class ActiveContoursFrame(ctk.CTkFrame):
    CANVAS_W = 400
    CANVAS_H = 400

    def __init__(self, master, app, **kwargs):
        super().__init__(master, **kwargs)
        self.app = app
        self.rect_coords = [50, 50, 200, 200]
        self.procesando = False
        self.seleccionando = False
        self.rect_id = None
        self.start_x = 0
        self.start_y = 0
        self.scale_x = 1.0
        self.scale_y = 1.0
        self._img_rgb = None
        self._photo_orig = None
        self._photo_seg = None
        self.build_ui()

    # ------------------------------------------------------------
    # UI BUILD
    # ------------------------------------------------------------
    def build_ui(self):
        controls = ctk.CTkFrame(self, width=280)
        controls.pack(side="left", fill="y", padx=10, pady=10, ipadx=10)
        controls.pack_propagate(False)

        ctk.CTkButton(controls, text="Cargar Imagen",
                      command=self.cargar_imagen).pack(pady=5, fill="x")

        ctk.CTkLabel(controls, text="Rectangulo Inicial (r0, c0, r1, c1)",
                     font=("Arial", 12)).pack(pady=(10, 0))
        frame_coords = ctk.CTkFrame(controls)
        frame_coords.pack(pady=5, fill="x")
        self.entry_r0 = ctk.CTkEntry(frame_coords, width=60,
                                      placeholder_text="r0")
        self.entry_r0.grid(row=0, column=0, padx=2)
        self.entry_c0 = ctk.CTkEntry(frame_coords, width=60,
                                      placeholder_text="c0")
        self.entry_c0.grid(row=0, column=1, padx=2)
        self.entry_r1 = ctk.CTkEntry(frame_coords, width=60,
                                      placeholder_text="r1")
        self.entry_r1.grid(row=0, column=2, padx=2)
        self.entry_c1 = ctk.CTkEntry(frame_coords, width=60,
                                      placeholder_text="c1")
        self.entry_c1.grid(row=0, column=3, padx=2)

        ctk.CTkButton(controls, text="Actualizar Rectangulo",
                      command=self.actualizar_rect).pack(pady=5, fill="x")

        ctk.CTkButton(controls, text="Seleccionar Rectangulo",
                      command=self.activar_seleccion).pack(pady=5, fill="x")

        ctk.CTkLabel(controls, text="Parametros de Color",
                     font=("Arial", 12)).pack(pady=(10, 0))
        ctk.CTkLabel(controls, text="Theta0 (Fondo):").pack(anchor="w")
        self.entry_t0 = ctk.CTkEntry(
            controls, placeholder_text="Ej: 30 (gris) o 100,150,200 (RGB)")
        self.entry_t0.pack(fill="x", pady=2)
        ctk.CTkLabel(controls, text="Theta1 (Objeto):").pack(anchor="w")
        self.entry_t1 = ctk.CTkEntry(
            controls, placeholder_text="Ej: 200 (gris) o 50,60,70 (RGB)")
        self.entry_t1.pack(fill="x", pady=2)

        ctk.CTkButton(controls, text="Estimar Parametros desde Rect",
                      command=self.estimar).pack(pady=5, fill="x")

        ctk.CTkLabel(controls, text="Max Iteraciones:").pack(anchor="w",
                                                              pady=(10, 0))
        self.entry_iter = ctk.CTkEntry(controls, placeholder_text="100")
        self.entry_iter.pack(fill="x", pady=2)
        self.entry_iter.insert(0, "100")

        self.btn_segmentar = ctk.CTkButton(
            controls, text="Segmentar", command=self.iniciar_segmentacion,
            fg_color="green")
        self.btn_segmentar.pack(pady=10, fill="x")

        self.label_info = ctk.CTkLabel(controls, text="Listo.",
                                       wraplength=250, justify="left")
        self.label_info.pack(pady=5)

        # Viewer panel
        viewer = ctk.CTkFrame(self)
        viewer.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        frame_orig = ctk.CTkFrame(viewer)
        frame_orig.pack(side="left", fill="both", expand=True, padx=5)
        ctk.CTkLabel(frame_orig, text="Imagen Original",
                     font=("Arial", 14)).pack()
        self.canvas_orig = tk.Canvas(
            frame_orig, width=self.CANVAS_W, height=self.CANVAS_H,
            bg="gray", cursor="arrow")
        self.canvas_orig.pack(expand=True, fill="both", padx=5, pady=5)
        self.canvas_orig.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas_orig.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas_orig.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.frame_hist_orig = ctk.CTkFrame(frame_orig, height=150)
        self.frame_hist_orig.pack(fill="x", pady=5)

        frame_seg = ctk.CTkFrame(viewer)
        frame_seg.pack(side="right", fill="both", expand=True, padx=5)
        ctk.CTkLabel(frame_seg, text="Segmentacion (Overlay Rojo)",
                     font=("Arial", 14)).pack()
        self.label_segmentada = ctk.CTkLabel(frame_seg, text="")
        self.label_segmentada.pack(expand=True, fill="both")
        self.frame_hist_seg = ctk.CTkFrame(frame_seg, height=150)
        self.frame_hist_seg.pack(fill="x", pady=5)

    # ------------------------------------------------------------
    # CANVAS IMAGE DISPLAY
    # ------------------------------------------------------------
    def mostrar_imagen_en_canvas(self, img_rgb):
        self._img_rgb = img_rgb
        h, w = img_rgb.shape[:2]
        cw = self.canvas_orig.winfo_width() or self.CANVAS_W
        ch = self.canvas_orig.winfo_height() or self.CANVAS_H
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)
        self.scale_x = w / nw
        self.scale_y = h / nh
        pil_img = Image.fromarray(img_rgb).resize((nw, nh), Image.LANCZOS)
        self._photo_orig = ImageTk.PhotoImage(pil_img)
        self.canvas_orig.delete("all")
        self.canvas_orig.create_image(cw // 2, ch // 2,
                                      image=self._photo_orig, anchor="center")
        self._draw_rect_on_canvas()

    def _draw_rect_on_canvas(self):
        self.canvas_orig.delete("rect")
        if not hasattr(self, '_img_rgb') or self._img_rgb is None:
            return
        r0, c0, r1, c1 = self.rect_coords
        x0 = c0 / self.scale_x
        y0 = r0 / self.scale_y
        x1 = c1 / self.scale_x
        y1 = r1 / self.scale_y
        cw = self.canvas_orig.winfo_width() or self.CANVAS_W
        ch = self.canvas_orig.winfo_height() or self.CANVAS_H
        h, w = self._img_rgb.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)
        ox = (cw - nw) // 2
        oy = (ch - nh) // 2
        self.canvas_orig.create_rectangle(
            ox + x0, oy + y0, ox + x1, oy + y1,
            outline="red", width=2, tags="rect")

    # ------------------------------------------------------------
    # MOUSE RECTANGLE SELECTION
    # ------------------------------------------------------------
    def activar_seleccion(self):
        if self._img_rgb is None:
            self.label_info.configure(text="Cargue una imagen primero.")
            return
        self.seleccionando = True
        self.canvas_orig.configure(cursor="cross")
        self.label_info.configure(text="Arrastre para seleccionar rectangulo.")

    def _canvas_to_image(self, cx, cy):
        cw = self.canvas_orig.winfo_width() or self.CANVAS_W
        ch = self.canvas_orig.winfo_height() or self.CANVAS_H
        h, w = self._img_rgb.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)
        ox = (cw - nw) // 2
        oy = (ch - nh) // 2
        ix = (cx - ox) * self.scale_x
        iy = (cy - oy) * self.scale_y
        return int(round(ix)), int(round(iy))

    def on_mouse_down(self, event):
        if not self.seleccionando:
            return
        self.start_x = event.x
        self.start_y = event.y
        if self.rect_id:
            self.canvas_orig.delete(self.rect_id)
        self.rect_id = self.canvas_orig.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline="red", width=2)

    def on_mouse_move(self, event):
        if not self.seleccionando or self.rect_id is None:
            return
        self.canvas_orig.coords(self.rect_id,
                                self.start_x, self.start_y, event.x, event.y)

    def on_mouse_up(self, event):
        if not self.seleccionando:
            return
        self.seleccionando = False
        self.canvas_orig.configure(cursor="arrow")
        if self.rect_id:
            self.canvas_orig.delete(self.rect_id)
            self.rect_id = None
        x1_img, y1_img = self._canvas_to_image(self.start_x, self.start_y)
        x2_img, y2_img = self._canvas_to_image(event.x, event.y)
        c0 = min(x1_img, x2_img)
        c1 = max(x1_img, x2_img)
        r0 = min(y1_img, y2_img)
        r1 = max(y1_img, y2_img)
        self.entry_r0.delete(0, ctk.END)
        self.entry_c0.delete(0, ctk.END)
        self.entry_r1.delete(0, ctk.END)
        self.entry_c1.delete(0, ctk.END)
        self.entry_r0.insert(0, str(r0))
        self.entry_c0.insert(0, str(c0))
        self.entry_r1.insert(0, str(r1))
        self.entry_c1.insert(0, str(c1))
        self.actualizar_rect()
        self._draw_rect_on_canvas()
        self.label_info.configure(text=f"Rectangulo seleccionado: ({r0},{c0})-({r1},{c1})")

    # ------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------
    def _plot_hist(self, img, parent, title):
        for w in parent.winfo_children():
            w.destroy()
        fig = Figure(figsize=(3, 1.5), dpi=100)
        ax = fig.add_subplot(111)
        if img is not None:
            hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 255))
            ax.bar(range(256), hist, width=1, color='gray')
            ax.set_title(title, fontsize=8)
            ax.set_xlim(0, 255)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # ------------------------------------------------------------
    # ACTIONS
    # ------------------------------------------------------------
    def cargar_imagen(self):
        path = filedialog.askopenfilename(
            filetypes=[("Imagenes", "*.png *.jpg *.jpeg *.bmp *.tif"),
                       ("Raw", "*.raw")])
        if not path:
            return
        if path.endswith('.raw'):
            try:
                img = np.fromfile(path, dtype=np.uint8).reshape(512, 512)
                color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                gray = img
            except Exception:
                self.label_info.configure(text="Error leyendo .raw")
                return
        else:
            img = cv2.imread(path)
            if img is None:
                return
            color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.app.current_image_color = color
        self.app.current_image = gray
        self.mostrar_imagen_en_canvas(color)
        self._plot_hist(gray, self.frame_hist_orig, "Histograma Original")
        self.label_info.configure(text="Imagen cargada.")

    def actualizar_rect(self):
        try:
            r0 = int(self.entry_r0.get())
            c0 = int(self.entry_c0.get())
            r1 = int(self.entry_r1.get())
            c1 = int(self.entry_c1.get())
            if r0 < 0 or c0 < 0 or r1 < r0 or c1 < c0:
                raise ValueError
            self.rect_coords = [r0, c0, r1, c1]
            self._draw_rect_on_canvas()
            self.label_info.configure(text=f"Rect: {self.rect_coords}")
        except Exception:
            self.label_info.configure(text="Coordenadas invalidas.")

    def estimar(self):
        if self.app.current_image_color is None:
            self.label_info.configure(text="Cargue una imagen primero.")
            return
        self.actualizar_rect()
        try:
            img = self.app.current_image_color
            theta0, theta1 = estimar_parametros(img,
                                                tuple(self.rect_coords))
            self.entry_t0.delete(0, ctk.END)
            self.entry_t1.delete(0, ctk.END)
            if isinstance(theta0, np.ndarray):
                self.entry_t0.insert(
                    0, f"{theta0[0]:.1f}, {theta0[1]:.1f}, {theta0[2]:.1f}")
                self.entry_t1.insert(
                    0, f"{theta1[0]:.1f}, {theta1[1]:.1f}, {theta1[2]:.1f}")
            else:
                self.entry_t0.insert(0, f"{theta0:.1f}")
                self.entry_t1.insert(0, f"{theta1:.1f}")
            self.label_info.configure(
                text="Parametros estimados desde el rectangulo.")
        except Exception as e:
            self.label_info.configure(text=f"Error estimando: {str(e)}")

    # ------------------------------------------------------------
    # SEGMENTATION (threaded)
    # ------------------------------------------------------------
    def iniciar_segmentacion(self):
        if self.procesando:
            return
        if self.app.current_image_color is None:
            self.label_info.configure(text="Cargue una imagen primero.")
            return
        self.procesando = True
        self.btn_segmentar.configure(state="disabled",
                                      text="Procesando...")
        self.label_info.configure(text="Segmentando en hilo secundario...")
        threading.Thread(target=self._ejecutar, daemon=True).start()

    def _ejecutar(self):
        try:
            self.actualizar_rect()
            H, W = self.app.current_image_color.shape[:2]
            r0, c0, r1, c1 = self.rect_coords
            if r0 <= 0 or r1 >= H - 1 or c0 <= 0 or c1 >= W - 1:
                self.app.after(
                    0, self._mostrar_error,
                    "Rectangulo pegado al borde. Ajuste para dejar espacio exterior.")
                return
            r0 = max(0, r0); c0 = max(0, c0)
            r1 = min(H - 1, r1); c1 = min(W - 1, c1)
            if r0 >= r1 or c0 >= c1:
                self.app.after(0, self._mostrar_error, "Rectangulo invalido.")
                return
            imagen = self.app.current_image_color

            def parse_theta(s):
                s = s.strip()
                if ',' in s:
                    return np.array([float(x.strip()) for x in s.split(',')])
                return float(s)

            theta0 = parse_theta(self.entry_t0.get())
            theta1 = parse_theta(self.entry_t1.get())
            max_iter = int(self.entry_iter.get())

            mascara, phi, L_in, L_out, iters = segmentar_imagen(
                imagen, tuple(self.rect_coords), theta0, theta1, max_iter)

            overlay = imagen.copy()
            if overlay.ndim == 2:
                overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
            bordes = (phi == -1) | (phi == 1)
            overlay[bordes] = [255, 0, 0]

            self.app.after(0, self._mostrar_resultado, overlay, iters,
                           phi, mascara)
        except Exception as e:
            self.app.after(0, self._mostrar_error, str(e))

    def _mostrar_resultado(self, overlay, iters, phi, mascara):
        ctk_img = convert_cv_to_ctk(overlay)
        self.label_segmentada.configure(image=ctk_img)
        self.label_segmentada.image = ctk_img
        self._plot_hist(mascara, self.frame_hist_seg,
                        "Mascara Segmentada")
        self.label_info.configure(
            text=f"Segmentado en {iters} iteraciones. "
                 f"Pixeles objeto: {np.sum(mascara > 0)}")
        self.procesando = False
        self.btn_segmentar.configure(state="normal", text="Segmentar")

    def _mostrar_error(self, msg):
        self.label_info.configure(text=f"Error: {msg}")
        self.procesando = False
        self.btn_segmentar.configure(state="normal", text="Segmentar")
