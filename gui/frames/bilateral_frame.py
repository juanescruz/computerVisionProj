"""Bilateral frame: bilateral filter vs anisotropic diffusion."""
import customtkinter as ctk

from gui.utils import convert_cv_to_ctk


class BilateralFrame(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.noisy_image = None
        self.original_image = None

        self.title_label = ctk.CTkLabel(
            self, text="Filtro Bilateral vs Difusión Anisotrópica",
            font=("Arial", 16, "bold")
        )
        self.title_label.pack(pady=5)

        # Bilateral controls
        ctrl = ctk.CTkFrame(self)
        ctrl.pack(pady=3)

        ctk.CTkLabel(ctrl, text="σ espacial:").pack(side="left", padx=2)
        self.ss_var = ctk.IntVar(value=5)
        self.ss_slider = ctk.CTkSlider(ctrl, from_=1, to=30, number_of_steps=29,
                                        variable=self.ss_var,
                                        command=self._on_ss_change, width=100)
        self.ss_slider.pack(side="left", padx=2)
        self.ss_label = ctk.CTkLabel(ctrl, text="5", width=25)
        self.ss_label.pack(side="left", padx=2)

        ctk.CTkLabel(ctrl, text="  σ rango:").pack(side="left", padx=(10, 2))
        self.sr_var = ctk.IntVar(value=50)
        self.sr_slider = ctk.CTkSlider(ctrl, from_=10, to=200, number_of_steps=190,
                                        variable=self.sr_var,
                                        command=self._on_sr_change, width=100)
        self.sr_slider.pack(side="left", padx=2)
        self.sr_label = ctk.CTkLabel(ctrl, text="50", width=30)
        self.sr_label.pack(side="left", padx=2)

        # Diffusion controls
        df = ctk.CTkFrame(self)
        df.pack(pady=3)

        ctk.CTkLabel(df, text="Dif. anis. iteraciones:").pack(side="left", padx=2)
        self.di_var = ctk.IntVar(value=20)
        self.di_slider = ctk.CTkSlider(df, from_=1, to=50, number_of_steps=49,
                                        variable=self.di_var,
                                        command=self._on_di_change, width=100)
        self.di_slider.pack(side="left", padx=2)
        self.di_label = ctk.CTkLabel(df, text="20", width=25)
        self.di_label.pack(side="left", padx=2)

        ctk.CTkLabel(df, text="  λ:").pack(side="left", padx=(10, 2))
        self.dl_var = ctk.DoubleVar(value=0.25)
        self.dl_slider = ctk.CTkSlider(df, from_=0.01, to=0.25, number_of_steps=24,
                                        variable=self.dl_var,
                                        command=self._on_dl_change, width=80)
        self.dl_slider.pack(side="left", padx=2)
        self.dl_label = ctk.CTkLabel(df, text="0.25", width=35)
        self.dl_label.pack(side="left", padx=2)

        ctk.CTkLabel(df, text="  k:").pack(side="left", padx=(10, 2))
        self.dk_var = ctk.IntVar(value=20)
        self.dk_slider = ctk.CTkSlider(df, from_=5, to=50, number_of_steps=45,
                                        variable=self.dk_var,
                                        command=self._on_dk_change, width=80)
        self.dk_slider.pack(side="left", padx=2)
        self.dk_label = ctk.CTkLabel(df, text="20", width=25)
        self.dk_label.pack(side="left", padx=2)

        # Noise controls
        nf = ctk.CTkFrame(self)
        nf.pack(pady=3, fill="x")

        ctk.CTkLabel(nf, text="Ruido:").pack(side="left", padx=5)
        self.noise_var = ctk.StringVar(value="Gaussiano")
        self.noise_menu = ctk.CTkOptionMenu(
            nf, variable=self.noise_var, values=["Gaussiano", "Sal y Pimienta"], command=self.on_noise_change
        )
        self.noise_menu.pack(side="left", padx=5)

        ctk.CTkLabel(nf, text="  %:").pack(side="left", padx=(10, 2))
        self.pct_var = ctk.IntVar(value=10)
        self.pct_slider = ctk.CTkSlider(nf, from_=0, to=100,
                                         variable=self.pct_var,
                                         command=self._on_pct_change, width=100)
        self.pct_slider.pack(side="left", padx=2)
        self.pct_label = ctk.CTkLabel(nf, text="10%", width=30)
        self.pct_label.pack(side="left", padx=2)
        
        self.mean_label = ctk.CTkLabel(nf, text="μ (mean):")
        self.mean_label.pack(side="left", padx=(10, 2))
        self.mean_entry = ctk.CTkEntry(nf, width=50)
        self.mean_entry.insert(0, "0")
        self.mean_entry.pack(side="left", padx=2)
        
        self.on_noise_change(self.noise_var.get())

        ctk.CTkLabel(nf, text="  σ/p:").pack(side="left", padx=(10, 2))
        self.noise_param_entry = ctk.CTkEntry(nf, width=50)
        self.noise_param_entry.insert(0, "25")
        self.noise_param_entry.pack(side="left", padx=2)

        self.btn_noise = ctk.CTkButton(nf, text="Aplicar Ruido",
                                       command=self.apply_noise, width=110)
        self.btn_noise.pack(side="left", padx=10)

        # Process button
        self.btn_process = ctk.CTkButton(self, text="Procesar y Comparar",
                                         command=self.process_all, width=200)
        self.btn_process.pack(pady=5)

        self.progress = ctk.CTkProgressBar(self, width=300)
        self.progress.pack(pady=2)
        self.progress.set(0)

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
            ("Difusión Anisotrópica", 1, 0), ("Filtro Bilateral", 1, 1),
        ]
        self._labels = {}
        for title, r, c in entries:
            f = ctk.CTkFrame(grid)
            f.grid(row=r, column=c, sticky="nsew", padx=4, pady=4)
            ctk.CTkLabel(f, text=title, font=("Arial", 11, "bold")).pack(pady=2)
            lbl = ctk.CTkLabel(f, text="Sin imagen", text_color="gray")
            lbl.pack(expand=True, pady=5)
            self._labels[title] = lbl

    # --- slider callbacks ---
    def _on_ss_change(self, v):
        self.ss_label.configure(text=str(int(v)))

    def _on_sr_change(self, v):
        self.sr_label.configure(text=str(int(v)))

    def _on_di_change(self, v):
        self.di_label.configure(text=str(int(v)))

    def _on_dl_change(self, v):
        self.dl_label.configure(text=f"{v:.2f}")

    def _on_dk_change(self, v):
        self.dk_label.configure(text=str(int(v)))

    def _on_pct_change(self, v):
        self.pct_label.configure(text=f"{int(v)}%")

    def on_noise_change(self, selection):
        if selection == "Gaussiano":
            self.mean_label.pack(side="left", padx=(10, 2))
            self.mean_entry.pack(side="left", padx=2)

        else:  # Sal y Pimienta
            self.mean_label.pack_forget()
            self.mean_entry.pack_forget()
            
            
    def _show(self, img, title):
        if img is None:
            return
        h, w = img.shape[:2]
        ctk_img = convert_cv_to_ctk(img, size=(min(w, 250), min(h, 200)))
        self._labels[title].configure(image=ctk_img, text="")
        self._labels[title].image = ctk_img

    def apply_noise(self):
        if self.app.current_image_color is not None:
            self.original_image = self.app.current_image_color.copy()
        else:
            self.original_image = self.app.current_image.copy()
        if self.original_image is None:
            self.status_label.configure(text="ERROR: No hay imagen", text_color="red")
            return

        noise_type = self.noise_var.get()
        pct = self.pct_slider.get() / 100.0
        try:
            mean = float(self.mean_entry.get() or "0")
            param = float(self.noise_param_entry.get() or "25")
        except ValueError:
            mean = 0.0
            param = 25.0

        try:
            from processing.noise_contamination import (
                add_gaussian_noise, add_salt_pepper_noise
            )
            if noise_type == "Gaussiano":
                mean = float(self.mean_entry.get() or "0")
                param  = float(self.noise_param_entry.get() or "25")
                self.noisy_image = add_gaussian_noise(
                    self.original_image, pct, mean, param
                )
            else:
                param = float(self.noise_param_entry.get() or "5")
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
        self._labels["Difusión Anisotrópica"].configure(
            image=None, text="Sin resultado", text_color="gray")
        self._labels["Filtro Bilateral"].configure(
            image=None, text="Sin resultado", text_color="gray")
        self.progress.set(0)
        self.status_label.configure(text=f"Ruido {noise_type} aplicado", text_color="green")

    def process_all(self):
        if self.noisy_image is None:
            self.status_label.configure(text="ERROR: Primero aplica ruido", text_color="red")
            return

        self.progress.set(0)
        self.status_label.configure(text="Procesando difusión...", text_color="orange")
        self.update_idletasks()

        try:
            from processing.anisotropic_diffusion import anisotropic_diffusion_color
            from processing.bilateral_filter import bilateral_filter

            diff_result = anisotropic_diffusion_color(
                self.noisy_image,
                iterations=int(self.di_slider.get()),
                lambda_param=self.dl_slider.get(),
                k=self.dk_slider.get(),
                diffusion_type="gaussian"
            )
            self._show(diff_result, "Difusión Anisotrópica")
            self.progress.set(0.5)
            self.update_idletasks()

            bilateral_result = bilateral_filter(
                self.noisy_image,
                sigma_spatial=self.ss_slider.get(),
                sigma_range=self.sr_slider.get(),
            )
            self._show(bilateral_result, "Filtro Bilateral")

            self._show(self.original_image, "Original")
            self._show(self.noisy_image, "Con Ruido")
            self.progress.set(1)
            self.status_label.configure(text="Procesamiento completado", text_color="green")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status_label.configure(text=f"ERROR: {str(e)}", text_color="red")

    def update_display(self):
        if self.app.current_image is None:
            for t in ["Original", "Con Ruido", "Difusión Anisotrópica", "Filtro Bilateral"]:
                self._labels[t].configure(image=None, text="Sin imagen", text_color="gray")
            self.noisy_image = None
            self.original_image = None
            self.status_label.configure(text="Sin imagen", text_color="gray")
            return
        if self.app.current_image_color is not None:
            self.original_image = self.app.current_image_color.copy()
        else:
            self.original_image = self.app.current_image.copy()
        self.noisy_image = None
        for t in ["Con Ruido", "Difusión Anisotrópica", "Filtro Bilateral"]:
            self._labels[t].configure(image=None, text="Sin resultado", text_color="gray")
        self._show(self.original_image, "Original")
        self.status_label.configure(text="Listo", text_color="green")
