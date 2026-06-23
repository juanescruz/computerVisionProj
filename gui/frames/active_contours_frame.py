"""Active contours (pixel exchange) segmentation frame with 2x2 grid and noise."""
import os
import tkinter as tk
import customtkinter as ctk
import numpy as np
import cv2
import threading
from tkinter import filedialog
from PIL import Image, ImageTk
from datetime import datetime
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from gui.utils import convert_cv_to_ctk
from processing.active_contours import segmentar_imagen, estimar_parametros
from processing.noise_contamination import (
    add_gaussian_noise, add_exponential_noise, add_rayleigh_noise,
    add_salt_pepper_noise
)


class ActiveContoursFrame(ctk.CTkFrame):
    CANVAS_W = 400
    CANVAS_H = 350

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
        self.imagen_original = None
        self.noisy_image = None
        self.imagen_actual = None
        self.overlay_original = None
        self.overlay_ruidosa = None
        self.mascara_original = None
        self.mascara_ruidosa = None
        self._iters_orig = 0
        self._iters_noisy = 0
        self._photo_orig = None
        self._dsp_ox = 0
        self._dsp_oy = 0
        self._dsp_nw = 0
        self._dsp_nh = 0
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

        self.entry_r0.insert(0, "50")
        self.entry_c0.insert(0, "50")
        self.entry_r1.insert(0, "200")
        self.entry_c1.insert(0, "200")

        ctk.CTkButton(controls, text="Actualizar Rectangulo",
                      command=self.actualizar_rect).pack(pady=5, fill="x")

        ctk.CTkButton(controls, text="Seleccionar Rectangulo",
                      command=self.activar_seleccion).pack(pady=5, fill="x")

        ctk.CTkLabel(controls, text="Ruido", font=("Arial", 12)).pack(pady=(10, 0))
        noise_top = ctk.CTkFrame(controls)
        noise_top.pack(fill="x", pady=2)
        self.noise_var = ctk.StringVar(value="Gaussiano")
        self.noise_menu = ctk.CTkOptionMenu(
            noise_top, variable=self.noise_var,
            values=["Gaussiano", "Exponencial", "Rayleigh", "Sal y Pimienta"],
            command=self._on_noise_change, width=120)
        self.noise_menu.pack(side="left", padx=2)

        noise_params = ctk.CTkFrame(controls)
        noise_params.pack(fill="x", pady=2)
        ctk.CTkLabel(noise_params, text="p1:").pack(side="left", padx=2)
        self.noise_p1 = ctk.CTkEntry(noise_params, width=55)
        self.noise_p1.insert(0, "0")
        self.noise_p1.pack(side="left", padx=2)
        ctk.CTkLabel(noise_params, text="p2:").pack(side="left", padx=(8, 2))
        self.noise_p2 = ctk.CTkEntry(noise_params, width=55)
        self.noise_p2.insert(0, "25")
        self.noise_p2.pack(side="left", padx=2)

        noise_pct_frame = ctk.CTkFrame(controls)
        noise_pct_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(noise_pct_frame, text="%:").pack(side="left", padx=2)
        self.noise_pct = ctk.CTkSlider(
            noise_pct_frame, from_=0, to=100, number_of_steps=100, width=100,
            command=lambda v: self.noise_pct_label.configure(text=f"{int(v)}%"))
        self.noise_pct.set(10)
        self.noise_pct.pack(side="left", padx=2)
        self.noise_pct_label = ctk.CTkLabel(noise_pct_frame, text="10%")
        self.noise_pct_label.pack(side="left", padx=2)

        noise_btns = ctk.CTkFrame(controls)
        noise_btns.pack(fill="x", pady=2)
        ctk.CTkButton(noise_btns, text="Aplicar Ruido",
                      command=self.aplicar_ruido, width=110).pack(side="left", padx=2)
        ctk.CTkButton(noise_btns, text="Restaurar",
                      command=self.restaurar_original, width=80).pack(side="left", padx=2)

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
        self.btn_segmentar.pack(pady=5, fill="x")

        ctk.CTkButton(controls, text="Exportar 4 Imagenes",
                      command=self.exportar_imagenes).pack(pady=5, fill="x")

        self.label_info = ctk.CTkLabel(controls, text="Listo.",
                                       wraplength=250, justify="left")
        self.label_info.pack(pady=5)

        # --------------------------------------------------------
        # 2x2 viewer grid
        # --------------------------------------------------------
        viewer = ctk.CTkFrame(self)
        viewer.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        viewer.grid_rowconfigure((0, 1), weight=1)
        viewer.grid_columnconfigure((0, 1), weight=1)

        # (0,0) Original — uses Canvas for rectangle selection
        f0 = ctk.CTkFrame(viewer)
        f0.grid(row=0, column=0, sticky="nsew", padx=3, pady=3)
        f0.grid_rowconfigure(0, weight=0)
        f0.grid_rowconfigure(1, weight=1)
        f0.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(f0, text="Original", font=("Arial", 12, "bold")).grid(row=0, column=0, pady=2)
        self.canvas_orig = tk.Canvas(
            f0, width=self.CANVAS_W, height=self.CANVAS_H,
            bg="gray", cursor="arrow", highlightthickness=0)
        self.canvas_orig.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
        self.canvas_orig.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas_orig.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas_orig.bind("<ButtonRelease-1>", self.on_mouse_up)

        # (0,1) Segmentada Original
        f1 = ctk.CTkFrame(viewer)
        f1.grid(row=0, column=1, sticky="nsew", padx=3, pady=3)
        f1.grid_rowconfigure(0, weight=0)
        f1.grid_rowconfigure(1, weight=1)
        f1.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(f1, text="Segmentada (Original)", font=("Arial", 12, "bold")).grid(row=0, column=0, pady=2)
        self.lbl_seg_orig = ctk.CTkLabel(f1, text="")
        self.lbl_seg_orig.grid(row=1, column=0, sticky="nsew")

        # (1,0) Contaminada
        f2 = ctk.CTkFrame(viewer)
        f2.grid(row=1, column=0, sticky="nsew", padx=3, pady=3)
        f2.grid_rowconfigure(0, weight=0)
        f2.grid_rowconfigure(1, weight=1)
        f2.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(f2, text="Contaminada", font=("Arial", 12, "bold")).grid(row=0, column=0, pady=2)
        self.lbl_noisy = ctk.CTkLabel(f2, text="")
        self.lbl_noisy.grid(row=1, column=0, sticky="nsew")

        # (1,1) Segmentada Contaminada
        f3 = ctk.CTkFrame(viewer)
        f3.grid(row=1, column=1, sticky="nsew", padx=3, pady=3)
        f3.grid_rowconfigure(0, weight=0)
        f3.grid_rowconfigure(1, weight=1)
        f3.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(f3, text="Segmentada (Contaminada)", font=("Arial", 12, "bold")).grid(row=0, column=0, pady=2)
        self.lbl_seg_noisy = ctk.CTkLabel(f3, text="")
        self.lbl_seg_noisy.grid(row=1, column=0, sticky="nsew")

    # ------------------------------------------------------------
    # CANVAS IMAGE DISPLAY
    # ------------------------------------------------------------
    def _mostrar_original_en_canvas(self, img_rgb):
        if img_rgb is None:
            self.canvas_orig.delete("all")
            return
        h, w = img_rgb.shape[:2]
        cw = max(self.canvas_orig.winfo_width(), self.CANVAS_W)
        ch = max(self.canvas_orig.winfo_height(), self.CANVAS_H)
        scale = min(cw / w, ch / h)
        self._dsp_nw = max(int(w * scale), 1)
        self._dsp_nh = max(int(h * scale), 1)
        self.scale_x = w / self._dsp_nw
        self.scale_y = h / self._dsp_nh
        self._dsp_ox = (cw - self._dsp_nw) // 2
        self._dsp_oy = (ch - self._dsp_nh) // 2
        pil_img = Image.fromarray(img_rgb).resize(
            (self._dsp_nw, self._dsp_nh), Image.LANCZOS)
        self._photo_orig = ImageTk.PhotoImage(pil_img)
        self.canvas_orig.delete("all")
        self.canvas_orig.create_image(
            self._dsp_ox + self._dsp_nw // 2,
            self._dsp_oy + self._dsp_nh // 2,
            image=self._photo_orig, anchor="center")
        self._draw_rect_on_canvas()

    def _draw_rect_on_canvas(self):
        self.canvas_orig.delete("rect_seleccion")
        if self.imagen_original is None:
            return
        r0, c0, r1, c1 = self.rect_coords
        x0 = self._dsp_ox + c0 / self.scale_x
        y0 = self._dsp_oy + r0 / self.scale_y
        x1 = self._dsp_ox + c1 / self.scale_x
        y1 = self._dsp_oy + r1 / self.scale_y
        self.canvas_orig.create_rectangle(
            x0, y0, x1, y1,
            outline="red", width=2, tags="rect_seleccion")

    # ------------------------------------------------------------
    # MOUSE RECTANGLE SELECTION
    # ------------------------------------------------------------
    def activar_seleccion(self):
        if self.imagen_original is None:
            self.label_info.configure(text="Cargue una imagen primero.")
            return
        self.seleccionando = True
        self.canvas_orig.configure(cursor="cross")
        self.label_info.configure(text="Arrastre para seleccionar rectangulo.")

    def _canvas_to_image(self, cx, cy):
        ix = (cx - self._dsp_ox) * self.scale_x
        iy = (cy - self._dsp_oy) * self.scale_y
        return max(0, int(round(ix))), max(0, int(round(iy)))

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
        h, w = self.imagen_original.shape[:2]
        c0 = max(0, min(x1_img, x2_img, w - 1))
        c1 = max(0, min(max(x1_img, x2_img), w - 1))
        r0 = max(0, min(y1_img, y2_img, h - 1))
        r1 = max(0, min(max(y1_img, y2_img), h - 1))
        if r0 >= r1:
            r1 = min(r0 + 1, h - 1)
        if c0 >= c1:
            c1 = min(c0 + 1, w - 1)
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
        self.label_info.configure(
            text=f"Rectangulo seleccionado: ({r0},{c0})-({r1},{c1})")

    # ------------------------------------------------------------
    # DISPLAY HELPERS
    # ------------------------------------------------------------
    def _img_to_label(self, lbl, img_rgb, label_size=(400, 280)):
        if img_rgb is None:
            lbl.configure(image=None, text="Sin imagen", text_color="gray")
            return
        ctk_img = convert_cv_to_ctk(img_rgb, size=label_size)
        lbl.configure(image=ctk_img, text="")
        lbl.image = ctk_img

    def _mostrar_todo(self):
        hs = (400, 280)
        self._mostrar_original_en_canvas(self.imagen_original)
        self._img_to_label(self.lbl_seg_orig, self.overlay_original, hs)
        self._img_to_label(self.lbl_noisy, self.noisy_image, hs)
        self._img_to_label(self.lbl_seg_noisy, self.overlay_ruidosa, hs)

    def _build_overlay(self, img_rgb, phi):
        overlay = img_rgb.copy()
        if overlay.ndim == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
        bordes = (phi == -1) | (phi == 1)
        overlay[bordes] = [255, 0, 0]
        return overlay

    # ------------------------------------------------------------
    # NOISE
    # ------------------------------------------------------------
    def _on_noise_change(self, selection):
        if selection == "Gaussiano":
            self.noise_p1.delete(0, ctk.END); self.noise_p1.insert(0, "0")
            self.noise_p2.configure(state="normal")
            self.noise_p2.delete(0, ctk.END); self.noise_p2.insert(0, "25")
        elif selection == "Exponencial":
            self.noise_p1.delete(0, ctk.END); self.noise_p1.insert(0, "0.05")
            self.noise_p2.configure(state="disabled")
        elif selection == "Rayleigh":
            self.noise_p1.delete(0, ctk.END); self.noise_p1.insert(0, "1.2")
            self.noise_p2.configure(state="disabled")
        elif selection == "Sal y Pimienta":
            self.noise_p1.delete(0, ctk.END); self.noise_p1.insert(0, "0.05")
            self.noise_p2.configure(state="disabled")

    def aplicar_ruido(self):
        if self.imagen_original is None:
            self.label_info.configure(text="Cargue una imagen primero.")
            return
        ntype = self.noise_var.get()
        pct = self.noise_pct.get() / 100.0
        try:
            if ntype == "Gaussiano":
                mean = float(self.noise_p1.get())
                sigma = float(self.noise_p2.get())
                noisy = add_gaussian_noise(self.imagen_original, pct, mean, sigma)
            elif ntype == "Exponencial":
                lam = float(self.noise_p1.get())
                noisy = add_exponential_noise(self.imagen_original, pct, lam)
            elif ntype == "Rayleigh":
                xi = float(self.noise_p1.get())
                noisy = add_rayleigh_noise(self.imagen_original, pct, xi)
            else:
                p = float(self.noise_p1.get())
                noisy = add_salt_pepper_noise(self.imagen_original, p)
        except Exception as e:
            self.label_info.configure(text=f"Error ruido: {e}")
            return
        self.noisy_image = noisy
        self.overlay_ruidosa = None
        self.mascara_ruidosa = None
        self._mostrar_todo()
        self.label_info.configure(text=f"Ruido {ntype} aplicado (color).")

    def restaurar_original(self):
        if self.imagen_original is None:
            return
        self.noisy_image = None
        self.overlay_ruidosa = None
        self.mascara_ruidosa = None
        self._mostrar_todo()
        self.label_info.configure(text="Imagen original restaurada.")

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
            except Exception:
                self.label_info.configure(text="Error leyendo .raw")
                return
        else:
            img = cv2.imread(path)
            if img is None:
                return
            color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.app.current_image_color = color
        self.app.current_image = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
        self.imagen_original = color.copy()
        self.noisy_image = None
        self.overlay_original = None
        self.overlay_ruidosa = None
        self.mascara_original = None
        self.mascara_ruidosa = None
        self._mostrar_todo()
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
        if self.imagen_original is None:
            self.label_info.configure(text="Cargue una imagen primero.")
            return
        self.actualizar_rect()
        try:
            theta0, theta1 = estimar_parametros(self.imagen_original,
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
        if self.imagen_original is None:
            self.label_info.configure(text="Cargue una imagen primero.")
            return
        # Read all GUI params on main thread
        try:
            r0 = int(self.entry_r0.get()); c0 = int(self.entry_c0.get())
            r1 = int(self.entry_r1.get()); c1 = int(self.entry_c1.get())
            self.rect_coords = [r0, c0, r1, c1]
            H, W = self.imagen_original.shape[:2]
            if r0 <= 0 or r1 >= H - 1 or c0 <= 0 or c1 >= W - 1:
                self.label_info.configure(
                    text="Rectangulo pegado al borde. Ajuste.")
                return
            if r0 >= r1 or c0 >= c1:
                self.label_info.configure(text="Rectangulo invalido.")
                return

            def parse_theta(s):
                s = s.strip()
                if ',' in s:
                    return np.array([float(x.strip()) for x in s.split(',')])
                return float(s)

            self._theta0 = parse_theta(self.entry_t0.get())
            self._theta1 = parse_theta(self.entry_t1.get())
            self._max_iter = int(self.entry_iter.get())
            self._rect = tuple(self.rect_coords)
        except Exception as e:
            self.label_info.configure(text=f"Parametros invalidos: {e}")
            return

        self.procesando = True
        self._result_ready = False
        self._result_error = None
        self.btn_segmentar.configure(state="disabled", text="Procesando...")
        self.label_info.configure(text="Segmentando...")
        threading.Thread(target=self._ejecutar, daemon=True).start()
        self.app.after(100, self._poll_result)

    def _ejecutar(self):
        try:
            masc_orig, phi_orig, _, _, it_orig = segmentar_imagen(
                self.imagen_original, self._rect,
                self._theta0, self._theta1, self._max_iter)
            ov_orig = self._build_overlay(self.imagen_original, phi_orig)
            ov_noisy = None
            if self.noisy_image is not None:
                masc_noisy, phi_noisy, _, _, it_noisy = segmentar_imagen(
                    self.noisy_image, self._rect,
                    self._theta0, self._theta1, self._max_iter)
                ov_noisy = self._build_overlay(self.noisy_image, phi_noisy)
                self._iters_noisy = it_noisy
                self.mascara_ruidosa = masc_noisy
            self._result = (ov_orig, masc_orig, it_orig, ov_noisy)
            self._result_ready = True
        except Exception as e:
            self._result_error = str(e)
            self._result_ready = True

    def _poll_result(self):
        if not self._result_ready:
            self.app.after(100, self._poll_result)
            return
        if self._result_error:
            self._mostrar_error(self._result_error)
            return
        ov_orig, masc_orig, it_orig, ov_noisy = self._result
        self.overlay_original = ov_orig
        self.mascara_original = masc_orig
        self._iters_orig = it_orig
        self.overlay_ruidosa = ov_noisy
        self._mostrar_resultado()

    def _mostrar_resultado(self):
        self._mostrar_todo()
        msg = f"Original: {self._iters_orig} iter."
        if self.overlay_ruidosa is not None:
            msg += f" | Contaminada: {self._iters_noisy} iter."
        self.label_info.configure(text=msg)
        self._guardar_resultados()
        self.procesando = False
        self.btn_segmentar.configure(state="normal", text="Segmentar")

    def _guardar_resultados(self):
        base = os.path.join("resultados", "ContornosActivos")
        os.makedirs(base, exist_ok=True)
        if self.imagen_original is not None:
            cv2.imwrite(os.path.join(base, "original.png"),
                        cv2.cvtColor(self.imagen_original, cv2.COLOR_RGB2BGR))
        if self.overlay_original is not None:
            cv2.imwrite(os.path.join(base, "segmentacion_original.png"),
                        cv2.cvtColor(self.overlay_original, cv2.COLOR_RGB2BGR))
        if self.noisy_image is not None:
            cv2.imwrite(os.path.join(base, "ruidosa.png"),
                        cv2.cvtColor(self.noisy_image, cv2.COLOR_RGB2BGR))
        if self.overlay_ruidosa is not None:
            cv2.imwrite(os.path.join(base, "segmentacion_ruidosa.png"),
                        cv2.cvtColor(self.overlay_ruidosa, cv2.COLOR_RGB2BGR))

    def _mostrar_error(self, msg):
        self.label_info.configure(text=f"Error: {msg}")
        self.procesando = False
        self.btn_segmentar.configure(state="normal", text="Segmentar")

    # ------------------------------------------------------------
    # EXPORT
    # ------------------------------------------------------------
    def exportar_imagenes(self):
        if self.imagen_original is None:
            self.label_info.configure(text="Cargue una imagen primero.")
            return
        base = os.path.join("assets", "exports")
        os.makedirs(base, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        cv2.imwrite(os.path.join(base, f"act_original_{ts}.png"),
                    cv2.cvtColor(self.imagen_original, cv2.COLOR_RGB2BGR))

        if self.overlay_original is not None:
            cv2.imwrite(os.path.join(base, f"act_segmentada_original_{ts}.png"),
                        cv2.cvtColor(self.overlay_original, cv2.COLOR_RGB2BGR))

        if self.noisy_image is not None:
            cv2.imwrite(os.path.join(base, f"act_ruidosa_{ts}.png"),
                        cv2.cvtColor(self.noisy_image, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(os.path.join(base, f"act_ruidosa_{ts}.png"),
                        cv2.cvtColor(self.imagen_original, cv2.COLOR_RGB2BGR))

        if self.overlay_ruidosa is not None:
            cv2.imwrite(os.path.join(base, f"act_segmentada_ruidosa_{ts}.png"),
                        cv2.cvtColor(self.overlay_ruidosa, cv2.COLOR_RGB2BGR))

        self.label_info.configure(text=f"Exportado a {base}/")
