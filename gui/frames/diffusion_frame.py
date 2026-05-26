"""Diffusion frame: isotropic / anisotropic vs median filter."""
import customtkinter as ctk

from gui.utils import convert_cv_to_ctk


class DiffusionFrame(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.noisy_image = None
        self.original_image = None

        self.title_label = ctk.CTkLabel(self, text="Difusión Isotrópica / Anisotrópica",
                                        font=("Arial", 16, "bold"))
        self.title_label.pack(pady=5)

        # Diffusion controls
        ctrl = ctk.CTkFrame(self)
        ctrl.pack(pady=3)

        ctk.CTkLabel(ctrl, text="Método:").pack(side="left", padx=5)
        self.method_var = ctk.StringVar(value="Isotrópica")
        self.method_menu = ctk.CTkOptionMenu(
            ctrl, variable=self.method_var,
            values=["Isotrópica", "Anisotrópica (Leclerc)", "Anisotrópica (Lorentz)"]
        )
        self.method_menu.pack(side="left", padx=5)

        ctk.CTkLabel(ctrl, text="  Iter:").pack(side="left", padx=(10, 2))
        self.iter_var = ctk.IntVar(value=20)
        self.iter_slider = ctk.CTkSlider(ctrl, from_=1, to=50, number_of_steps=49,
                                          variable=self.iter_var,
                                          command=self._on_iter_change, width=100)
        self.iter_slider.pack(side="left", padx=2)
        self.iter_label = ctk.CTkLabel(ctrl, text="20", width=25)
        self.iter_label.pack(side="left", padx=2)

        ctk.CTkLabel(ctrl, text="  λ:").pack(side="left", padx=(10, 2))
        self.lambda_var = ctk.DoubleVar(value=0.25)
        self.lambda_slider = ctk.CTkSlider(ctrl, from_=0.01, to=0.25, number_of_steps=24,
                                            variable=self.lambda_var,
                                            command=self._on_lambda_change, width=90)
        self.lambda_slider.pack(side="left", padx=2)
        self.lambda_label = ctk.CTkLabel(ctrl, text="0.25", width=30)
        self.lambda_label.pack(side="left", padx=2)

        ctk.CTkLabel(ctrl, text="  k:").pack(side="left", padx=(10, 2))
        self.k_var = ctk.IntVar(value=20)
        self.k_slider = ctk.CTkSlider(ctrl, from_=5, to=50, number_of_steps=45,
                                       variable=self.k_var,
                                       command=self._on_k_change, width=90)
        self.k_slider.pack(side="left", padx=2)
        self.k_label = ctk.CTkLabel(ctrl, text="20", width=25)
        self.k_label.pack(side="left", padx=2)

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
        self.pct_slider = ctk.CTkSlider(nf, from_=0, to=100,
                                         variable=self.pct_var,
                                         command=self._on_pct_change, width=100)
        self.pct_slider.pack(side="left", padx=2)
        self.pct_label = ctk.CTkLabel(nf, text="10%", width=30)
        self.pct_label.pack(side="left", padx=2)

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

        # 2x2 grid display
        grid = ctk.CTkFrame(self)
        grid.pack(fill="both", expand=True, padx=5, pady=5)
        for i in range(2):
            grid.grid_columnconfigure(i, weight=1, uniform="col")
            grid.grid_rowconfigure(i, weight=1)

        entries = [
            ("Original", 0, 0), ("Con Ruido", 0, 1),
            ("Filtro Mediana", 1, 0), ("Difusión", 1, 1),
        ]
        self._labels = {}
        for title, r, c in entries:
            f = ctk.CTkFrame(grid)
            f.grid(row=r, column=c, sticky="nsew", padx=4, pady=4)
            ctk.CTkLabel(f, text=title, font=("Arial", 11, "bold")).pack(pady=2)
            lbl = ctk.CTkLabel(f, text="Sin imagen", text_color="gray")
            lbl.pack(expand=True, pady=5)
            self._labels[title] = lbl

        self.status_label = ctk.CTkLabel(self, text="Listo", text_color="green")
        self.status_label.pack(pady=2)

    # --- slider callbacks ---
    def _on_iter_change(self, v):
        self.iter_label.configure(text=str(int(v)))

    def _on_lambda_change(self, v):
        self.lambda_label.configure(text=f"{v:.2f}")

    def _on_k_change(self, v):
        self.k_label.configure(text=str(int(v)))

    def _on_pct_change(self, v):
        self.pct_label.configure(text=f"{int(v)}%")

    def _show(self, img, title):
        if img is None:
            self.status_label.configure(text=f"ERROR: {title} es None", text_color="red")
            return
        h, w = img.shape[:2]
        ctk_img = convert_cv_to_ctk(img, size=(min(w, 250), min(h, 200)))
        self._labels[title].configure(image=ctk_img, text="")
        self._labels[title].image = ctk_img

    def apply_noise(self):
        self.original_image = self.app.get_current_image()
        if self.original_image is None:
            import tkinter.messagebox as msgbox
            msgbox.showwarning("Advertencia", "Cargar imagen primero desde Inicio.")
            self.status_label.configure(text="ERROR: No hay imagen cargada", text_color="red")
            return

        noise_type = self.noise_var.get()
        pct = self.pct_slider.get() / 100.0
        try:
            param = float(self.noise_param_entry.get() or "25")
        except ValueError:
            param = 25.0

        try:
            from processing.noise_contamination import add_gaussian_noise, add_salt_pepper_noise
            if noise_type == "Gaussiano":
                self.noisy_image = add_gaussian_noise(self.original_image, pct, 0, param)
            else:
                self.noisy_image = add_salt_pepper_noise(self.original_image, param / 100)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status_label.configure(text=f"Error ruido: {str(e)}", text_color="red")
            return

        self._show(self.original_image, "Original")
        self._show(self.noisy_image, "Con Ruido")
        self._labels["Filtro Mediana"].configure(image=None, text="Sin resultado", text_color="gray")
        self._labels["Difusión"].configure(image=None, text="Sin resultado", text_color="gray")
        self.status_label.configure(text=f"Ruido {noise_type} aplicado", text_color="green")

    def process_all(self):
        if self.noisy_image is None:
            import tkinter.messagebox as msgbox
            msgbox.showwarning("Advertencia", "Aplicar ruido primero.")
            self.status_label.configure(text="ERROR: Primero aplica ruido", text_color="red")
            return

        try:
            from processing.spatial_filters import median_filter
            from processing.anisotropic_diffusion import isotropic_diffusion, anisotropic_diffusion
        except ImportError as e:
            self.status_label.configure(text=f"ERROR import: {str(e)}", text_color="red")
            return

        method = self.method_var.get()
        iters = int(self.iter_slider.get())
        lmb = self.lambda_slider.get()
        k_val = self.k_slider.get()

        self.status_label.configure(text="Procesando...", text_color="orange")
        self.update_idletasks()

        try:
            median_result = median_filter(self.noisy_image, kernel_size=3)
            self._show(median_result, "Filtro Mediana")

            if method == "Isotrópica":
                diff_result = isotropic_diffusion(self.noisy_image, iters, lmb)
            else:
                diff_type = "gaussian" if "Gauss" in method else "lorentz"
                diff_result = anisotropic_diffusion(self.noisy_image, iters, lmb, k_val, diff_type)

            self._show(self.original_image, "Original")
            self._show(self.noisy_image, "Con Ruido")
            self._show(median_result, "Filtro Mediana")
            self._show(diff_result, "Difusión")
            self.status_label.configure(text=f"{method} completado ({iters} iter, λ={lmb:.2f})", text_color="green")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status_label.configure(text=f"ERROR procesamiento: {str(e)}", text_color="red")

    def update_display(self):
        if self.app.current_image is None:
            for t in ["Original", "Con Ruido", "Filtro Mediana", "Difusión"]:
                self._labels[t].configure(image=None, text="Sin imagen", text_color="gray")
            self.noisy_image = None
            self.original_image = None
            self.status_label.configure(text="Sin imagen", text_color="gray")
            return
        self.original_image = self.app.get_current_image()
        self.noisy_image = None
        for t in ["Con Ruido", "Filtro Mediana", "Difusión"]:
            self._labels[t].configure(image=None, text="Sin resultado", text_color="gray")
        self._show(self.original_image, "Original")
        self.status_label.configure(text="Listo", text_color="green")
