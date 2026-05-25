"""Frame combining noise contamination and edge detection."""
import customtkinter as ctk
import numpy as np

from processing.noise_contamination import (
    add_gaussian_noise, add_exponential_noise, add_rayleigh_noise,
    add_salt_pepper_noise
)
from processing.edge_detection import prewitt_operator, sobel_operator
from gui.utils import convert_cv_to_ctk, plot_histogram


class EdgeNoiseFrame(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.noisy_image = None
        self._edge_result = None
        self._hist_canvases = [None, None, None]

        self.title_label = ctk.CTkLabel(self, text="Bordes en Imágenes con Ruido",
                                        font=("Arial", 16, "bold"))
        self.title_label.pack(pady=5)

        self.controls_frame = ctk.CTkFrame(self)
        self.controls_frame.pack(pady=5, fill="x")

        self.noise_label = ctk.CTkLabel(self.controls_frame, text="Tipo ruido:")
        self.noise_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.noise_var = ctk.StringVar(value="Gaussiano")
        self.noise_menu = ctk.CTkOptionMenu(
            self.controls_frame, variable=self.noise_var,
            values=["Gaussiano", "Exponencial", "Rayleigh", "Sal y Pimienta"],
            command=self.on_noise_change
        )
        self.noise_menu.grid(row=0, column=1, padx=5, pady=5)

        self.param1_label = ctk.CTkLabel(self.controls_frame, text="Media (μ):")
        self.param1_label.grid(row=0, column=2, padx=(15, 5), pady=5, sticky="w")
        self.param1_entry = ctk.CTkEntry(self.controls_frame, width=80)
        self.param1_entry.insert(0, "0")
        self.param1_entry.grid(row=0, column=3, padx=5, pady=5)

        self.param2_label = ctk.CTkLabel(self.controls_frame, text="Sigma (σ):")
        self.param2_label.grid(row=0, column=4, padx=(15, 5), pady=5, sticky="w")
        self.param2_entry = ctk.CTkEntry(self.controls_frame, width=80)
        self.param2_entry.insert(0, "25")
        self.param2_entry.grid(row=0, column=5, padx=5, pady=5)

        self.percent_label = ctk.CTkLabel(self.controls_frame, text="%:")
        self.percent_label.grid(row=0, column=6, padx=(15, 5), pady=5)

        self.percent_var = ctk.IntVar(value=10)
        self.percent_slider = ctk.CTkSlider(
            self.controls_frame, from_=0, to=100,
            variable=self.percent_var, command=self.on_percent_change, width=120
        )
        self.percent_slider.grid(row=0, column=7, padx=5, pady=5)

        self.percent_value_label = ctk.CTkLabel(self.controls_frame, text="10%")
        self.percent_value_label.grid(row=0, column=8, padx=5, pady=5)

        self.btn_noise = ctk.CTkButton(
            self.controls_frame, text="Aplicar Ruido",
            command=self.apply_noise, width=120
        )
        self.btn_noise.grid(row=0, column=9, padx=5, pady=5)

        self.edge_frame = ctk.CTkFrame(self)
        self.edge_frame.pack(pady=5, fill="x")

        ctk.CTkLabel(self.edge_frame, text="Operador:").pack(side="left", padx=5)
        self.operator_var = ctk.StringVar(value="Sobel")
        self.operator_menu = ctk.CTkOptionMenu(
            self.edge_frame, variable=self.operator_var,
            values=["Prewitt", "Sobel"]
        )
        self.operator_menu.pack(side="left", padx=5)

        self.btn_detect = ctk.CTkButton(
            self.edge_frame, text="Detectar Bordes",
            command=self.detect_edges, width=150
        )
        self.btn_detect.pack(side="left", padx=5)

        self.btn_reset = ctk.CTkButton(
            self.edge_frame, text="Reset",
            command=self.reset_all, width=100
        )
        self.btn_reset.pack(side="left", padx=5)

        self.info_label = ctk.CTkLabel(self.edge_frame, text="")
        self.info_label.pack(side="left", padx=15)

        self.display_frame = ctk.CTkFrame(self)
        self.display_frame.pack(fill="both", expand=True, pady=5)
        self.display_frame.grid_columnconfigure((0, 1, 2), weight=1, uniform="col")
        self.display_frame.grid_rowconfigure(0, weight=1)

        cols = [
            ("Original", "Sin imagen"),
            ("Con Ruido", ""),
            ("Bordes", "Sin bordes"),
        ]
        self._image_labels = []
        self._hist_frames = []
        for c, (title, placeholder) in enumerate(cols):
            frame = ctk.CTkFrame(self.display_frame)
            frame.grid(row=0, column=c, sticky="nsew", padx=4)

            lbl = ctk.CTkLabel(frame, text=placeholder, text_color="gray")
            lbl.pack(pady=5)
            self._image_labels.append(lbl)

            ctk.CTkLabel(frame, text=title, font=("Arial", 12)).pack(pady=2)

            hf = ctk.CTkFrame(frame)
            hf.pack(fill="both", expand=True, pady=5)
            self._hist_frames.append(hf)

    def on_noise_change(self, selection):
        if selection == "Gaussiano":
            self.param1_label.configure(text="Media (μ):")
            self.param1_entry.delete(0, "end"); self.param1_entry.insert(0, "0")
            self.param2_label.configure(text="Sigma (σ):")
            self.param2_entry.delete(0, "end"); self.param2_entry.insert(0, "25")
            self.param2_label.grid(); self.param2_entry.grid()
            self.percent_label.grid(); self.percent_slider.grid(); self.percent_value_label.grid()
        elif selection == "Exponencial":
            self.param1_label.configure(text="Lambda (λ):")
            self.param1_entry.delete(0, "end"); self.param1_entry.insert(0, "0.05")
            self.param2_label.grid_remove(); self.param2_entry.grid_remove()
            self.percent_label.grid(); self.percent_slider.grid(); self.percent_value_label.grid()
        elif selection == "Rayleigh":
            self.param1_label.configure(text="Xi (ξ):")
            self.param1_entry.delete(0, "end"); self.param1_entry.insert(0, "1.2")
            self.param2_label.grid_remove(); self.param2_entry.grid_remove()
            self.percent_label.grid(); self.percent_slider.grid(); self.percent_value_label.grid()
        elif selection == "Sal y Pimienta":
            self.param1_label.configure(text="Prob. p:")
            self.param1_entry.delete(0, "end"); self.param1_entry.insert(0, "0.05")
            self.param2_label.grid_remove(); self.param2_entry.grid_remove()
            self.percent_label.grid_remove(); self.percent_slider.grid_remove(); self.percent_value_label.grid_remove()

    def on_percent_change(self, value):
        self.percent_value_label.configure(text=f"{int(value)}%")

    def _destroy_hist(self, idx):
        if self._hist_canvases[idx] is not None:
            self._hist_canvases[idx].get_tk_widget().destroy()
            self._hist_canvases[idx] = None

    def _place_hist(self, img, idx):
        self._destroy_hist(idx)
        canvas = plot_histogram(img, self._hist_frames[idx], width=260, height=200)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._hist_canvases[idx] = canvas

    def _show_image(self, img, idx):
        if img is None:
            return
        h, w = img.shape
        ctk_img = convert_cv_to_ctk(img, size=(min(w, 260), min(h, 200)))
        self._image_labels[idx].configure(image=ctk_img, text="")
        self._image_labels[idx].image = ctk_img
        self._place_hist(img, idx)

    def apply_noise(self):
        if self.app.current_image is None:
            import tkinter.messagebox as msgbox
            msgbox.showwarning("Advertencia", "Cargar una imagen primero desde Inicio.")
            return

        self._edge_result = None
        self._destroy_hist(2)
        self._image_labels[2].configure(image=None, text="Sin bordes", text_color="gray")

        noise_type = self.noise_var.get()
        pct = self.percent_slider.get() / 100.0

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

        self._show_image(self.app.current_image, 0)
        self._show_image(self.noisy_image, 1)
        self.info_label.configure(text=f"Ruido {noise_type} aplicado")

    def detect_edges(self):
        if self.noisy_image is None:
            import tkinter.messagebox as msgbox
            msgbox.showwarning("Advertencia", "Aplicar ruido primero.")
            return

        operator = self.operator_var.get()
        if operator == "Prewitt":
            self._edge_result = prewitt_operator(self.noisy_image)
        else:
            self._edge_result = sobel_operator(self.noisy_image)

        self._show_image(self._edge_result, 2)
        self.info_label.configure(text=f"{operator} sobre imagen con ruido")

    def reset_all(self):
        self.noisy_image = None
        self._edge_result = None
        self._destroy_hist(1)
        self._destroy_hist(2)
        self._image_labels[1].configure(image=None, text="Sin ruido", text_color="gray")
        self._image_labels[2].configure(image=None, text="Sin bordes", text_color="gray")
        self.info_label.configure(text="")

    def update_display(self):
        if self.app.current_image is None:
            for i in range(3):
                self._destroy_hist(i)
                self._image_labels[i].configure(image=None, text="Sin imagen", text_color="gray")
            self.noisy_image = None
            self._edge_result = None
            return

        self.noisy_image = None
        self._edge_result = None
        for i in range(1, 3):
            self._destroy_hist(i)
            self._image_labels[i].configure(image=None, text="Sin resultado", text_color="gray")
        self._show_image(self.app.current_image, 0)
