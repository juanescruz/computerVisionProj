"""Laplacian / LoG edge detection with noise comparison."""
import customtkinter as ctk
import numpy as np

from processing.noise_contamination import (
    add_gaussian_noise, add_exponential_noise, add_rayleigh_noise,
    add_salt_pepper_noise
)
from processing.edge_detection import (
    laplacian_zero_crossings, laplacian_with_slope, log_edge,
)
from gui.utils import convert_cv_to_ctk


class LaplacianFrame(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.noisy_image = None

        self.title_label = ctk.CTkLabel(self, text="Laplaciano / LoG con Ruido",
                                        font=("Arial", 16, "bold"))
        self.title_label.pack(pady=5)

        # Method
        method_frame = ctk.CTkFrame(self)
        method_frame.pack(pady=3)

        ctk.CTkLabel(method_frame, text="Método:").pack(side="left", padx=5)
        self.method_var = ctk.StringVar(value="Laplaciano")
        self.method_menu = ctk.CTkOptionMenu(
            method_frame, variable=self.method_var,
            values=["Laplaciano", "Laplaciano+Pendiente", "LoG"],
            command=self.on_method_change
        )
        self.method_menu.pack(side="left", padx=5)

        ctk.CTkLabel(method_frame, text="  σ:").pack(side="left", padx=(10, 2))
        self.sigma_var = ctk.DoubleVar(value=1.0)
        self.sigma_slider = ctk.CTkSlider(
            method_frame, from_=0.5, to=5.0, number_of_steps=90,
            variable=self.sigma_var, command=self.on_sigma_change, width=120
        )
        self.sigma_slider.pack(side="left", padx=2)
        self.sigma_label = ctk.CTkLabel(method_frame, text="1.0", width=30)
        self.sigma_label.pack(side="left", padx=2)

        ctk.CTkLabel(method_frame, text="  Umbral:").pack(side="left", padx=(10, 2))
        self.thresh_var = ctk.IntVar(value=30)
        self.thresh_slider = ctk.CTkSlider(
            method_frame, from_=0, to=100,
            variable=self.thresh_var, command=self.on_thresh_change, width=120
        )
        self.thresh_slider.pack(side="left", padx=2)
        self.thresh_label = ctk.CTkLabel(method_frame, text="30", width=30)
        self.thresh_label.pack(side="left", padx=2)

        # Noise controls
        noise_frame = ctk.CTkFrame(self)
        noise_frame.pack(pady=3, fill="x")

        ctk.CTkLabel(noise_frame, text="Ruido:").pack(side="left", padx=5)
        self.noise_var = ctk.StringVar(value="Gaussiano")
        self.noise_menu = ctk.CTkOptionMenu(
            noise_frame, variable=self.noise_var,
            values=["Gaussiano", "Exponencial", "Rayleigh", "Sal y Pimienta"],
            command=self.on_noise_change
        )
        self.noise_menu.pack(side="left", padx=5)

        self.param1_label = ctk.CTkLabel(noise_frame, text="μ:")
        self.param1_label.pack(side="left", padx=(10, 2))
        self.param1_entry = ctk.CTkEntry(noise_frame, width=60)
        self.param1_entry.insert(0, "0")
        self.param1_entry.pack(side="left", padx=2)

        self.param2_label = ctk.CTkLabel(noise_frame, text="σ:")
        self.param2_label.pack(side="left", padx=(5, 2))
        self.param2_entry = ctk.CTkEntry(noise_frame, width=60)
        self.param2_entry.insert(0, "25")
        self.param2_entry.pack(side="left", padx=2)

        ctk.CTkLabel(noise_frame, text="  %:").pack(side="left", padx=(10, 2))
        self.pct_var = ctk.IntVar(value=10)
        self.pct_slider = ctk.CTkSlider(
            noise_frame, from_=0, to=100,
            variable=self.pct_var, command=self.on_pct_change, width=120
        )
        self.pct_slider.pack(side="left", padx=2)
        self.pct_label = ctk.CTkLabel(noise_frame, text="10%", width=30)
        self.pct_label.pack(side="left", padx=2)

        self.btn_noise = ctk.CTkButton(noise_frame, text="Aplicar Ruido",
                                       command=self.apply_noise, width=120)
        self.btn_noise.pack(side="left", padx=10)

        # Detect button
        self.btn_detect = ctk.CTkButton(self, text="Detectar Bordes",
                                        command=self.detect_edges, width=200)
        self.btn_detect.pack(pady=5)

        # 2x2 grid display
        self.grid_frame = ctk.CTkFrame(self)
        self.grid_frame.pack(fill="both", expand=True, padx=5, pady=5)
        for i in range(2):
            self.grid_frame.grid_columnconfigure(i, weight=1, uniform="col")
            self.grid_frame.grid_rowconfigure(i, weight=1)

        labels = [
            ("Original", 0, 0), ("Con Ruido", 0, 1),
            ("Bordes (limpia)", 1, 0), ("Bordes (ruidosa)", 1, 1),
        ]
        self._img_labels = {}
        for text, r, c in labels:
            f = ctk.CTkFrame(self.grid_frame)
            f.grid(row=r, column=c, sticky="nsew", padx=4, pady=4)
            lbl = ctk.CTkLabel(f, text="Sin imagen", text_color="gray")
            lbl.pack(expand=True, pady=5)
            ctk.CTkLabel(f, text=text, font=("Arial", 11)).pack(pady=2)
            self._img_labels[(r, c)] = lbl

        self.info_label = ctk.CTkLabel(self, text="")
        self.info_label.pack(pady=3)

        self.on_method_change("Laplaciano")
        self.on_noise_change("Gaussiano")

    def on_method_change(self, sel):
        show_sigma = sel == "LoG"
        show_thresh = sel in ("Laplaciano+Pendiente", "LoG")
        for w in [self.sigma_slider, self.sigma_label]:
            w.pack() if show_sigma else w.pack_forget()
        for w in [self.thresh_slider, self.thresh_label]:
            w.pack() if show_thresh else w.pack_forget()

    def on_sigma_change(self, v):
        self.sigma_label.configure(text=f"{v:.1f}")

    def on_thresh_change(self, v):
        self.thresh_label.configure(text=str(int(v)))

    def on_noise_change(self, sel):
        if sel == "Gaussiano":
            self.param1_label.configure(text="μ:"); self.param1_entry.delete(0, "end"); self.param1_entry.insert(0, "0")
            self.param2_label.configure(text="σ:"); self.param2_entry.delete(0, "end"); self.param2_entry.insert(0, "25")
            self.param2_label.pack(side="left", padx=(5, 2)); self.param2_entry.pack(side="left", padx=2)
        elif sel == "Exponencial":
            self.param1_label.configure(text="λ:"); self.param1_entry.delete(0, "end"); self.param1_entry.insert(0, "0.05")
            self.param2_label.pack_forget(); self.param2_entry.pack_forget()
        elif sel == "Rayleigh":
            self.param1_label.configure(text="ξ:"); self.param1_entry.delete(0, "end"); self.param1_entry.insert(0, "1.2")
            self.param2_label.pack_forget(); self.param2_entry.pack_forget()
        elif sel == "Sal y Pimienta":
            self.param1_label.configure(text="p:"); self.param1_entry.delete(0, "end"); self.param1_entry.insert(0, "0.05")
            self.param2_label.pack_forget(); self.param2_entry.pack_forget()

    def on_pct_change(self, v):
        self.pct_label.configure(text=f"{int(v)}%")

    def _show_img(self, img, r, c):
        if img is None:
            return
        h, w = img.shape[:2]
        ctk_img = convert_cv_to_ctk(img, size=(min(w, 260), min(h, 200)))
        self._img_labels[(r, c)].configure(image=ctk_img, text="")
        self._img_labels[(r, c)].image = ctk_img

    def apply_noise(self):
        if self.app.current_image is None:
            import tkinter.messagebox as msgbox
            msgbox.showwarning("Advertencia", "Cargar una imagen primero desde Inicio.")
            return

        noise_type = self.noise_var.get()
        pct = self.pct_slider.get() / 100.0

        try:
            if noise_type == "Gaussiano":
                mean = float(self.param1_entry.get())
                sigma = float(self.param2_entry.get())
                self.noisy_image = add_gaussian_noise(self.app.current_image, pct, mean, sigma)
            elif noise_type == "Exponencial":
                lam = float(self.param1_entry.get())
                self.noisy_image = add_exponential_noise(self.app.current_image, pct, lam)
            elif noise_type == "Rayleigh":
                xi = float(self.param1_entry.get())
                self.noisy_image = add_rayleigh_noise(self.app.current_image, pct, xi)
            elif noise_type == "Sal y Pimienta":
                p = float(self.param1_entry.get())
                self.noisy_image = add_salt_pepper_noise(self.app.current_image, p)
        except Exception as e:
            self.info_label.configure(text=f"Error ruido: {str(e)}")
            return

        self._show_img(self.app.current_image, 0, 0)
        self._show_img(self.noisy_image, 0, 1)
        self.info_label.configure(text=f"Ruido {noise_type} aplicado")

    def _detect(self, img):
        method = self.method_var.get()
        if img is None:
            return None
        if method == "Laplaciano":
            return laplacian_zero_crossings(img)
        elif method == "Laplaciano+Pendiente":
            return laplacian_with_slope(img, self.thresh_var.get())
        else:
            return log_edge(img, self.sigma_var.get(), self.thresh_var.get())

    def detect_edges(self):
        if self.app.current_image is None:
            import tkinter.messagebox as msgbox
            msgbox.showwarning("Advertencia", "Cargar una imagen primero desde Inicio.")
            return

        clean_edges = self._detect(self.app.current_image)
        noisy_edges = self._detect(self.noisy_image)

        self._show_img(clean_edges, 1, 0)
        self._show_img(noisy_edges, 1, 1)
        self.info_label.configure(text=f"{self.method_var.get()} aplicado")

    def update_display(self):
        if self.app.current_image is None:
            for r, c in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                self._img_labels[(r, c)].configure(image=None, text="Sin imagen", text_color="gray")
            self.noisy_image = None
            return
        self.noisy_image = None
        for r, c in [(0, 1), (1, 0), (1, 1)]:
            self._img_labels[(r, c)].configure(image=None, text="Sin resultado", text_color="gray")
        self._show_img(self.app.current_image, 0, 0)
