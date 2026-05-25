"""Automatic thresholding frame: iterative, Otsu, Otsu per RGB band."""
import customtkinter as ctk
import numpy as np

from gui.utils import convert_cv_to_ctk


class ThresholdingAutoFrame(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.original_image = None
        self.noisy_image = None

        self.title_label = ctk.CTkLabel(
            self, text="Umbralización Automática",
            font=("Arial", 16, "bold")
        )
        self.title_label.pack(pady=5)

        # Method controls
        ctrl = ctk.CTkFrame(self)
        ctrl.pack(pady=3)

        ctk.CTkLabel(ctrl, text="Método:").pack(side="left", padx=5)
        self.method_var = ctk.StringVar(value="Iterativo")
        self.method_menu = ctk.CTkOptionMenu(
            ctrl, variable=self.method_var,
            values=["Iterativo", "Otsu", "Otsu por Bandas (Color)"]
        )
        self.method_menu.pack(side="left", padx=5)

        ctk.CTkLabel(ctrl, text="  ΔT:").pack(side="left", padx=(10, 2))
        self.delta_entry = ctk.CTkEntry(ctrl, width=50)
        self.delta_entry.insert(0, "1")
        self.delta_entry.pack(side="left", padx=2)

        self.threshold_display = ctk.CTkLabel(ctrl, text="Umbral: --", width=200)
        self.threshold_display.pack(side="left", padx=10)

        # Noise controls
        nf = ctk.CTkFrame(self)
        nf.pack(pady=3, fill="x")

        ctk.CTkLabel(nf, text="Ruido:").pack(side="left", padx=5)
        self.noise_var = ctk.StringVar(value="Gaussiano")
        self.noise_menu = ctk.CTkOptionMenu(
            nf, variable=self.noise_var, values=["Gaussiano", "Sal y Pimienta"]
        )
        self.noise_menu.pack(side="left", padx=5)

        ctk.CTkLabel(nf, text="  %:").pack(side="left", padx=(10, 2))
        self.pct_var = ctk.IntVar(value=10)
        self.pct_slider = ctk.CTkSlider(
            nf, from_=0, to=100, variable=self.pct_var,
            command=self._on_pct_change, width=100
        )
        self.pct_slider.pack(side="left", padx=2)
        self.pct_label = ctk.CTkLabel(nf, text="10%", width=30)
        self.pct_label.pack(side="left", padx=2)

        ctk.CTkLabel(nf, text="  σ/p:").pack(side="left", padx=(10, 2))
        self.noise_param_entry = ctk.CTkEntry(nf, width=50)
        self.noise_param_entry.insert(0, "25")
        self.noise_param_entry.pack(side="left", padx=2)

        self.btn_noise = ctk.CTkButton(
            nf, text="Aplicar Ruido", command=self.apply_noise, width=110
        )
        self.btn_noise.pack(side="left", padx=10)

        # Process button
        self.btn_process = ctk.CTkButton(
            self, text="Calcular y Umbralizar",
            command=self.process_all, width=200
        )
        self.btn_process.pack(pady=5)

        # Status
        self.status_label = ctk.CTkLabel(self, text="Listo", text_color="green")
        self.status_label.pack(pady=2)

        # 2x2 grid
        grid = ctk.CTkFrame(self)
        grid.pack(fill="both", expand=True, padx=5, pady=5)
        for i in range(2):
            grid.grid_columnconfigure(i, weight=1, uniform="col")
            grid.grid_rowconfigure(i, weight=1)

        entries = [
            ("Original", 0, 0), ("Con Ruido", 0, 1),
            ("Umbral (limpia)", 1, 0), ("Umbral (ruidosa)", 1, 1),
        ]
        self._labels = {}
        for title, r, c in entries:
            f = ctk.CTkFrame(grid)
            f.grid(row=r, column=c, sticky="nsew", padx=4, pady=4)
            ctk.CTkLabel(f, text=title, font=("Arial", 11, "bold")).pack(pady=2)
            lbl = ctk.CTkLabel(f, text="Sin imagen", text_color="gray")
            lbl.pack(expand=True, pady=5)
            self._labels[title] = lbl

    def _on_pct_change(self, v):
        self.pct_label.configure(text=f"{int(v)}%")

    def _show(self, img, title):
        if img is None:
            return
        h, w = img.shape[:2]
        ctk_img = convert_cv_to_ctk(img, size=(min(w, 250), min(h, 200)))
        self._labels[title].configure(image=ctk_img, text="")
        self._labels[title].image = ctk_img

    def apply_noise(self):
        if self.app.current_image is None:
            self.status_label.configure(text="ERROR: No hay imagen", text_color="red")
            return

        if self.app.current_image_color is not None:
            self.original_image = self.app.current_image_color.copy()
        else:
            self.original_image = self.app.current_image.copy()
        noise_type = self.noise_var.get()
        pct = self.pct_slider.get() / 100.0
        try:
            param = float(self.noise_param_entry.get() or "25")
        except ValueError:
            param = 25.0

        try:
            from processing.noise_contamination import (
                add_gaussian_noise, add_salt_pepper_noise
            )
            if noise_type == "Gaussiano":
                self.noisy_image = add_gaussian_noise(
                    self.original_image, pct, 0, param
                )
            else:
                self.noisy_image = add_salt_pepper_noise(
                    self.original_image, param / 100
                )
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status_label.configure(text=f"Error ruido: {str(e)}", text_color="red")
            return

        self._show(self.original_image, "Original")
        self._show(self.noisy_image, "Con Ruido")
        for t in ["Umbral (limpia)", "Umbral (ruidosa)"]:
            self._labels[t].configure(image=None, text="Sin resultado", text_color="gray")
        self.status_label.configure(text=f"Ruido {noise_type} aplicado", text_color="green")

    def process_all(self):
        if self.original_image is None:
            self.status_label.configure(text="ERROR: No hay imagen cargada", text_color="red")
            return

        self.status_label.configure(text="Procesando...", text_color="orange")
        self.update_idletasks()

        try:
            from processing.thresholding import (
                iterative_threshold, otsu_threshold, otsu_rgb_segmentation
            )

            method = self.method_var.get()
            try:
                delta = float(self.delta_entry.get() or "1")
            except ValueError:
                delta = 1.0

            # Select source image based on method
            if method == "Otsu por Bandas (Color)" and len(self.original_image.shape) == 2:
                clean = np.stack([self.original_image] * 3, axis=2)
                noisy = (
                    np.stack([self.noisy_image] * 3, axis=2)
                    if self.noisy_image is not None and len(self.noisy_image.shape) == 2
                    else self.noisy_image
                )
            else:
                clean = self.original_image
                noisy = self.noisy_image

            if method == "Iterativo":
                t_clean, bin_clean = iterative_threshold(clean, delta)
            elif method == "Otsu":
                t_clean, bin_clean = otsu_threshold(clean)
            else:
                bin_clean = otsu_rgb_segmentation(clean)
                t_clean = "RGB"

            if noisy is not None:
                if method == "Iterativo":
                    t_noisy, bin_noisy = iterative_threshold(noisy, delta)
                elif method == "Otsu":
                    t_noisy, bin_noisy = otsu_threshold(noisy)
                else:
                    bin_noisy = otsu_rgb_segmentation(noisy)
                    t_noisy = "RGB"
            else:
                bin_noisy = None
                t_noisy = "--"

            self.threshold_display.configure(
                text=f"Umbral: limpia={t_clean} | ruidosa={t_noisy}"
            )

            self._show(bin_clean, "Umbral (limpia)")
            if bin_noisy is not None:
                self._show(bin_noisy, "Umbral (ruidosa)")
            self.status_label.configure(
                text=f"{method} completado", text_color="green"
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status_label.configure(text=f"ERROR: {str(e)}", text_color="red")

    def update_display(self):
        if self.app.current_image is None:
            for t in ["Original", "Con Ruido", "Umbral (limpia)", "Umbral (ruidosa)"]:
                self._labels[t].configure(image=None, text="Sin imagen", text_color="gray")
            self.original_image = None
            self.noisy_image = None
            self.status_label.configure(text="Sin imagen", text_color="gray")
            self.threshold_display.configure(text="Umbral: --")
            return
        if self.app.current_image_color is not None:
            self.original_image = self.app.current_image_color.copy()
        else:
            self.original_image = self.app.current_image.copy()
        self.noisy_image = None
        for t in ["Con Ruido", "Umbral (limpia)", "Umbral (ruidosa)"]:
            self._labels[t].configure(image=None, text="Sin resultado", text_color="gray")
        self._show(self.original_image, "Original")
        self.status_label.configure(text="Listo", text_color="green")
        self.threshold_display.configure(text="Umbral: --")
