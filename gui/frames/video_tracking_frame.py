"""Video tracking frame using pixel exchange algorithm (Shi & Karl, 2005)."""
import tkinter as tk
import customtkinter as ctk
import numpy as np
import cv2
import threading
from tkinter import filedialog
from PIL import Image, ImageTk
from gui.utils import convert_cv_to_ctk
from processing.active_contours import (
    inicializar_contornos,
    calcular_Fd,
    actualizar_contornos,
    verificar_convergencia,
    estimar_parametros
)


class VideoTrackingFrame(ctk.CTkFrame):
    CANVAS_W = 400
    CANVAS_H = 400

    def __init__(self, master, app, **kwargs):
        super().__init__(master, **kwargs)
        self.app = app
        self.frames = []
        self.phis = []
        self.L_ins = []
        self.L_outs = []
        self.current_idx = 0
        self.rect_inicial = (50, 50, 150, 150)
        self.procesando = False
        self.seleccionando = False
        self.rect_id = None
        self.rect_seleccion_id = None
        self.start_x = 0
        self.start_y = 0
        self.scale_x = 1.0
        self.scale_y = 1.0
        self._photo = None
        self.build_ui()

    # ------------------------------------------------------------
    # UI BUILD
    # ------------------------------------------------------------
    def build_ui(self):
        controls = ctk.CTkFrame(self, width=280)
        controls.pack(side="left", fill="y", padx=10, pady=10, ipadx=10)
        controls.pack_propagate(False)

        ctk.CTkButton(controls, text="Cargar Video",
                      command=self.cargar_video).pack(pady=5, fill="x")

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
        ctk.CTkButton(controls, text="Actualizar Rect",
                      command=self.actualizar_rect).pack(pady=5, fill="x")

        ctk.CTkButton(controls, text="Seleccionar Rectangulo",
                      command=self.activar_seleccion).pack(pady=5, fill="x")

        ctk.CTkLabel(controls, text="Theta0 (Fondo):").pack(anchor="w",
                                                             pady=(10, 0))
        self.entry_t0 = ctk.CTkEntry(
            controls, placeholder_text="Ej: 30")
        self.entry_t0.pack(fill="x", pady=2)
        ctk.CTkLabel(controls, text="Theta1 (Objeto):").pack(anchor="w")
        self.entry_t1 = ctk.CTkEntry(
            controls, placeholder_text="Ej: 200")
        self.entry_t1.pack(fill="x", pady=2)
        ctk.CTkButton(controls, text="Estimar Parametros (desde 1er frame)",
                      command=self.estimar).pack(pady=5, fill="x")

        ctk.CTkLabel(controls, text="Max Iter por frame:").pack(anchor="w",
                                                                 pady=(10, 0))
        self.entry_iter = ctk.CTkEntry(controls, placeholder_text="50")
        self.entry_iter.pack(fill="x", pady=2)
        self.entry_iter.insert(0, "50")

        self.btn_seguir = ctk.CTkButton(
            controls, text="Iniciar Seguimiento",
            command=self.iniciar_seguimiento, fg_color="green")
        self.btn_seguir.pack(pady=10, fill="x")

        self.label_info = ctk.CTkLabel(controls, text="Listo.",
                                       wraplength=250, justify="left")
        self.label_info.pack(pady=5)

        self.slider = ctk.CTkSlider(controls, from_=0, to=0,
                                     command=self.actualizar_frame)
        self.slider.pack(fill="x", pady=10)
        self.label_frame_num = ctk.CTkLabel(controls, text="Frame: 0 / 0")
        self.label_frame_num.pack()

        viewer = ctk.CTkFrame(self)
        viewer.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        ctk.CTkLabel(viewer, text="Seguimiento (Overlay Rojo)",
                     font=("Arial", 14)).pack()
        self.canvas = tk.Canvas(
            viewer, width=self.CANVAS_W, height=self.CANVAS_H,
            bg="gray", cursor="arrow")
        self.canvas.pack(expand=True, fill="both", padx=5, pady=5)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        self.entry_r0.insert(0, "50")
        self.entry_c0.insert(0, "50")
        self.entry_r1.insert(0, "150")
        self.entry_c1.insert(0, "150")

    # ------------------------------------------------------------
    # CANVAS IMAGE DISPLAY
    # ------------------------------------------------------------
    def mostrar_en_canvas(self, img_rgb):
        h, w = img_rgb.shape[:2]
        cw = self.canvas.winfo_width() or self.CANVAS_W
        ch = self.canvas.winfo_height() or self.CANVAS_H
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)
        self.scale_x = w / nw
        self.scale_y = h / nh
        pil_img = Image.fromarray(img_rgb).resize((nw, nh), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2,
                                 image=self._photo, anchor="center")

    def _draw_rect_on_canvas(self):
        self.canvas.delete("rect_seleccion")
        if not self.frames:
            return
        r0, c0, r1, c1 = self.rect_inicial
        cw = self.canvas.winfo_width() or self.CANVAS_W
        ch = self.canvas.winfo_height() or self.CANVAS_H
        h, w = self.frames[0].shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)
        ox = (cw - nw) // 2
        oy = (ch - nh) // 2
        x0 = c0 / self.scale_x
        y0 = r0 / self.scale_y
        x1 = c1 / self.scale_x
        y1 = r1 / self.scale_y
        self.rect_seleccion_id = self.canvas.create_rectangle(
            ox + x0, oy + y0, ox + x1, oy + y1,
            outline="red", width=2, tags="rect_seleccion")

    # ------------------------------------------------------------
    # MOUSE RECTANGLE SELECTION
    # ------------------------------------------------------------
    def activar_seleccion(self):
        if not self.frames:
            self.label_info.configure(text="Cargue un video primero.")
            return
        self.seleccionando = True
        self.canvas.configure(cursor="cross")
        self.label_info.configure(text="Arrastre para seleccionar rectangulo.")

    def _canvas_to_image(self, cx, cy):
        cw = self.canvas.winfo_width() or self.CANVAS_W
        ch = self.canvas.winfo_height() or self.CANVAS_H
        h, w = self.frames[0].shape[:2]
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
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline="red", width=2)

    def on_mouse_move(self, event):
        if not self.seleccionando or self.rect_id is None:
            return
        self.canvas.coords(self.rect_id,
                           self.start_x, self.start_y, event.x, event.y)

    def on_mouse_up(self, event):
        if not self.seleccionando:
            return
        self.seleccionando = False
        self.canvas.configure(cursor="arrow")
        if self.rect_id:
            self.canvas.delete(self.rect_id)
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
        self.label_info.configure(
            text=f"Rectangulo seleccionado: ({r0},{c0})-({r1},{c1})")

    # ------------------------------------------------------------
    # LOAD & CONTROL
    # ------------------------------------------------------------
    def cargar_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if not path:
            return
        cap = cv2.VideoCapture(path)
        self.frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if self.frames:
            self.slider.configure(to=len(self.frames) - 1)
            self.phis = [None] * len(self.frames)
            self.L_ins = [None] * len(self.frames)
            self.L_outs = [None] * len(self.frames)
            self.current_idx = 0
            self.rect_seleccion_id = None
            self.actualizar_frame(0)
            self.label_info.configure(
                text=f"Video cargado: {len(self.frames)} frames")

    def actualizar_rect(self):
        try:
            r0 = int(self.entry_r0.get())
            c0 = int(self.entry_c0.get())
            r1 = int(self.entry_r1.get())
            c1 = int(self.entry_c1.get())
            if r0 < 0 or c0 < 0 or r1 < r0 or c1 < c0:
                raise ValueError
            self.rect_inicial = (r0, c0, r1, c1)
            self._draw_rect_on_canvas()
            self.label_info.configure(text=f"Rect: {self.rect_inicial}")
        except Exception:
            self.label_info.configure(text="Coordenadas invalidas.")

    def estimar(self):
        if not self.frames:
            self.label_info.configure(text="Cargue un video primero.")
            return
        self.actualizar_rect()
        try:
            img = self.frames[0]
            theta0, theta1 = estimar_parametros(img, self.rect_inicial)
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
                text="Parametros estimados desde el primer frame.")
        except Exception as e:
            self.label_info.configure(text=f"Error estimando: {str(e)}")

    def actualizar_frame(self, val):
        self.current_idx = int(val)
        if not self.frames:
            return
        img = self.frames[self.current_idx].copy()
        if self.phis and self.current_idx < len(self.phis) and \
                self.phis[self.current_idx] is not None:
            phi = self.phis[self.current_idx]
            bordes = (phi == -1) | (phi == 1)
            img[bordes] = [255, 0, 0]
        self.mostrar_en_canvas(img)
        self.label_frame_num.configure(
            text=f"Frame: {self.current_idx + 1} / {len(self.frames)}")

    # ------------------------------------------------------------
    # TRACKING WITH STATE PROPAGATION (p. 29)
    # ------------------------------------------------------------
    def iniciar_seguimiento(self):
        if self.procesando:
            return
        if not self.frames:
            self.label_info.configure(text="Cargue un video primero.")
            return
        if self.rect_seleccion_id is not None:
            self.canvas.delete(self.rect_seleccion_id)
            self.rect_seleccion_id = None
        self.procesando = True
        self.btn_seguir.configure(state="disabled", text="Procesando...")
        self.label_info.configure(text="Procesando seguimiento...")
        threading.Thread(target=self.ejecutar_seguimiento, daemon=True).start()

    def ejecutar_seguimiento(self):
        try:
            r0 = int(self.entry_r0.get()); c0 = int(self.entry_c0.get())
            r1 = int(self.entry_r1.get()); c1 = int(self.entry_c1.get())
            self.rect_inicial = (r0, c0, r1, c1)
            theta0_str = self.entry_t0.get().strip()
            theta1_str = self.entry_t1.get().strip()
            max_iter = int(self.entry_iter.get())

            def parse_theta(s):
                if ',' in s:
                    return np.array([float(x.strip())
                                    for x in s.split(',')])
                return float(s)

            theta0 = parse_theta(theta0_str)
            theta1 = parse_theta(theta1_str)
            H, W = self.frames[0].shape[:2]

            # Frame 0: initialise from rectangle
            L_in, L_out, phi = inicializar_contornos(
                self.rect_inicial, (H, W))
            Fd = calcular_Fd(self.frames[0], theta0, theta1)
            for _ in range(max_iter):
                L_in, L_out, phi = actualizar_contornos(
                    L_in, L_out, phi, Fd)
                if verificar_convergencia(L_in, L_out, Fd):
                    break

            self.phis[0] = phi.copy()
            self.L_ins[0] = L_in.copy()
            self.L_outs[0] = L_out.copy()
            self.app.after(0, self.actualizar_slider_y_frame, 0)

            # Frames 1..N: propagate state (L_in, L_out, phi) from previous frame
            for idx in range(1, len(self.frames)):
                Fd = calcular_Fd(self.frames[idx], theta0, theta1)
                for _ in range(max_iter):
                    L_in, L_out, phi = actualizar_contornos(
                        L_in, L_out, phi, Fd)
                    if verificar_convergencia(L_in, L_out, Fd):
                        break

                self.phis[idx] = phi.copy()
                self.L_ins[idx] = L_in.copy()
                self.L_outs[idx] = L_out.copy()

                if idx % 5 == 0 or idx == len(self.frames) - 1:
                    self.app.after(0, self.actualizar_slider_y_frame, idx)

            self.app.after(0, self.finalizar_seguimiento)

        except Exception as e:
            self.app.after(0, self.mostrar_error, str(e))

    # ------------------------------------------------------------
    # UI UPDATES (thread-safe)
    # ------------------------------------------------------------
    def actualizar_slider_y_frame(self, idx):
        self.slider.set(idx)
        self.actualizar_frame(idx)
        self.label_info.configure(
            text=f"Procesando frame {idx + 1}/{len(self.frames)}")

    def finalizar_seguimiento(self):
        self.label_info.configure(text="Seguimiento completado.")
        self.procesando = False
        self.btn_seguir.configure(state="normal", text="Iniciar Seguimiento")

    def mostrar_error(self, msg):
        self.label_info.configure(text=f"Error: {msg}")
        self.procesando = False
        self.btn_seguir.configure(state="normal", text="Iniciar Seguimiento")
