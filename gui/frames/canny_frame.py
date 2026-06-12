"""Dedicated Canny frame with noise and 4-way comparison."""
import customtkinter as ctk
import numpy as np

from processing.advanced_edge_detection import canny_edge_detector
from processing.noise_contamination import (
    add_gaussian_noise, add_exponential_noise, add_rayleigh_noise,
    add_salt_pepper_noise
)
from gui.utils import convert_cv_to_ctk


class CannyFrame(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.noisy_image = None
        self._canny_original = None
        self._canny_noisy = None

        self.title_label = ctk.CTkLabel(
            self, text="Detector de Bordes — Canny",
            font=("Arial", 16, "bold")
        )
        self.title_label.pack(pady=5)

        # ── Noise controls ──
        self.noise_frame = ctk.CTkFrame(self)
        self.noise_frame.pack(pady=2, fill="x")

        ctk.CTkLabel(self.noise_frame, text="Ruido:").pack(side="left", padx=5)
        self.noise_var = ctk.StringVar(value="Gaussiano")
        self.noise_menu = ctk.CTkOptionMenu(
            self.noise_frame, variable=self.noise_var,
            values=["Gaussiano", "Exponencial", "Rayleigh", "Sal y Pimienta"],
            command=self.on_noise_change, width=120
        )
        self.noise_menu.pack(side="left", padx=2)

        ctk.CTkLabel(self.noise_frame, text="μ/λ/ξ:").pack(side="left", padx=(8, 2))
        self.noise_p1 = ctk.CTkEntry(self.noise_frame, width=60)
        self.noise_p1.insert(0, "0")
        self.noise_p1.pack(side="left", padx=2)

        ctk.CTkLabel(self.noise_frame, text="σ:").pack(side="left", padx=(8, 2))
        self.noise_p2 = ctk.CTkEntry(self.noise_frame, width=60)
        self.noise_p2.insert(0, "25")
        self.noise_p2.pack(side="left", padx=2)

        ctk.CTkLabel(self.noise_frame, text="%:").pack(side="left", padx=(8, 2))
        self.noise_pct = ctk.CTkSlider(
            self.noise_frame,
            from_=0,
            to=100,
            number_of_steps=100,
            width=100,
            command=lambda v: self.noise_pct_label.configure(
                text=f"{int(v)}%"
            )
        )
        self.noise_pct.set(10)
        self.noise_pct.pack(side="left", padx=2)
        self.noise_pct_label = ctk.CTkLabel(self.noise_frame, text="10%")
        self.noise_pct_label.pack(side="left", padx=2)

        self.btn_apply_noise = ctk.CTkButton(
            self.noise_frame, text="Aplicar Ruido",
            command=self.apply_noise, width=120
        )
        self.btn_apply_noise.pack(side="left", padx=10)

        # ── Canny controls ──
        self.canny_control_frame = ctk.CTkFrame(self)
        self.canny_control_frame.pack(pady=2, fill="x")

        ctk.CTkLabel(self.canny_control_frame, text="Canny σ:").pack(side="left", padx=5)
        self.sigma_entry = ctk.CTkEntry(self.canny_control_frame, width=50)
        self.sigma_entry.insert(0, "1.0")
        self.sigma_entry.pack(side="left", padx=2)

        ctk.CTkLabel(self.canny_control_frame, text="Low:").pack(side="left", padx=(10, 2))
        self.low_entry = ctk.CTkEntry(self.canny_control_frame, width=50)
        self.low_entry.insert(0, "50")
        self.low_entry.pack(side="left", padx=2)

        ctk.CTkLabel(self.canny_control_frame, text="High:").pack(side="left", padx=(10, 2))
        self.high_entry = ctk.CTkEntry(self.canny_control_frame, width=50)
        self.high_entry.insert(0, "150")
        self.high_entry.pack(side="left", padx=2)

        self.btn_apply_canny = ctk.CTkButton(
            self.canny_control_frame, text="Aplicar Canny",
            command=self.apply_canny, width=140
        )
        self.btn_apply_canny.pack(side="left", padx=(15, 5))

        self.btn_reset = ctk.CTkButton(
            self.canny_control_frame, text="Reset", command=self.reset_all, width=100
        )
        self.btn_reset.pack(side="left", padx=5)

        self.info_label = ctk.CTkLabel(self.canny_control_frame, text="", text_color="gray")
        self.info_label.pack(side="left", padx=10)

        # ── 2x2 grid display ──
        self.grid_frame = ctk.CTkFrame(self)
        self.grid_frame.pack(fill="both", expand=True, pady=5)
        self.grid_frame.grid_rowconfigure((0, 1), weight=1)
        self.grid_frame.grid_columnconfigure((0, 1), weight=1)

        # positions: (row, col, label_text)
        self._panels = {}
        for r, c, key, title in [
            (0, 0, "orig", "Original"),
            (0, 1, "noisy", "Com Ruído"),
            (1, 0, "canny_orig", "Canny — Original"),
            (1, 1, "canny_noisy", "Canny — Com Ruído"),
        ]:
            panel = ctk.CTkFrame(self.grid_frame)
            panel.grid(row=r, column=c, sticky="nsew", padx=3, pady=3)
            panel.grid_rowconfigure(0, weight=1)
            panel.grid_columnconfigure(0, weight=1)

            lbl = ctk.CTkLabel(panel, text="Sem imagem", text_color="gray")
            lbl.grid(row=0, column=0, pady=3)
            ctk.CTkLabel(panel, text=title, font=("Arial", 11, "bold")).grid(row=1, column=0, pady=1)

            self._panels[key] = {"label": lbl}

    # ── Noise helpers ──
    def on_noise_change(self, selection):
        show_p2 = selection == "Gaussiano"
        for w in [self.noise_p2]:
            w.pack_info = w.pack_info() if hasattr(w, 'pack_info') else {}
            if show_p2:
                w.pack(side="left", padx=2)
            else:
                w.pack_forget()
        # simpler: just always show both, adjust labels
        if selection == "Gaussiano":
            self.noise_p1.delete(0, "end"); self.noise_p1.insert(0, "0")
            self.noise_p2.delete(0, "end"); self.noise_p2.insert(0, "25")
        elif selection == "Exponencial":
            self.noise_p1.delete(0, "end"); self.noise_p1.insert(0, "0.05")
        elif selection == "Rayleigh":
            self.noise_p1.delete(0, "end"); self.noise_p1.insert(0, "1.2")
        elif selection == "Sal y Pimienta":
            self.noise_p1.delete(0, "end"); self.noise_p1.insert(0, "0.05")

    def apply_noise(self):
        if self.app.current_image is None:
            return
        ntype = self.noise_var.get()
        pct = self.noise_pct.get() / 100.0
        try:
            if ntype == "Gaussiano":
                mean = float(self.noise_p1.get())
                sigma = float(self.noise_p2.get())
                self.noisy_image = add_gaussian_noise(self.app.current_image, pct, mean, sigma)
            elif ntype == "Exponencial":
                lam = float(self.noise_p1.get())
                self.noisy_image = add_exponential_noise(self.app.current_image, pct, lam)
            elif ntype == "Rayleigh":
                xi = float(self.noise_p1.get())
                self.noisy_image = add_rayleigh_noise(self.app.current_image, pct, xi)
            elif ntype == "Sal y Pimienta":
                p = float(self.noise_p1.get())
                self.noisy_image = add_salt_pepper_noise(self.app.current_image, p)
        except Exception as e:
            self.info_label.configure(text=f"Erro ruído: {e}", text_color="red")
            return
        self._canny_noisy = None
        self._show_images()
        self.info_label.configure(text=f"Ruido {ntype} aplicado", text_color="green")

    # ── Canny helpers ──
    def _get_canny_params(self):
        try:
            sigma = float(self.sigma_entry.get())
            low = float(self.low_entry.get())
            high = float(self.high_entry.get())
        except ValueError:
            return None
        sigma = max(0.3, min(3.0, sigma))
        low = max(0, min(255, low))
        high = max(0, min(255, high))
        if low > high:
            low, high = high, low
            self.low_entry.delete(0, "end"); self.low_entry.insert(0, str(int(low)))
            self.high_entry.delete(0, "end"); self.high_entry.insert(0, str(int(high)))
        return sigma, low, high

    def apply_canny(self):
        if self.app.current_image is None:
            return
        params = self._get_canny_params()
        if params is None:
            self.info_label.configure(text="Parâmetros inválidos", text_color="red")
            return
        sigma, low, high = params

        self._canny_original = canny_edge_detector(
            self.app.current_image, sigma, low, high
        )
        if self.noisy_image is not None:
            self._canny_noisy = canny_edge_detector(
                self.noisy_image, sigma, low, high
            )

        self._show_images()
        self.info_label.configure(
            text=f"Canny σ={sigma:.1f}  Low={low:.0f}  High={high:.0f}",
            text_color="green"
        )

    def reset_all(self):
        self.noisy_image = None
        self._canny_original = None
        self._canny_noisy = None
        self.info_label.configure(text="")
        self._show_images()

    # ── Display ──
    def _show_image_in_panel(self, img, key):
        panel = self._panels[key]
        if img is None:
            panel["label"].configure(image=None, text="Sem imagem", text_color="gray")
            return

        h, w = img.shape[:2]
        ctk_img = convert_cv_to_ctk(img, size=(min(w, 280), min(h, 220)))
        panel["label"].configure(image=ctk_img, text="")
        panel["label"].image = ctk_img

    def _show_images(self):
        self._show_image_in_panel(self.app.current_image, "orig")
        self._show_image_in_panel(self._canny_original, "canny_orig")
        self._show_image_in_panel(self.noisy_image, "noisy")
        self._show_image_in_panel(self._canny_noisy, "canny_noisy")

    def update_display(self):
        if self.app.current_image is None:
            for k in self._panels:
                self._show_image_in_panel(None, k)
            self.noisy_image = None
            self._canny_original = None
            self._canny_noisy = None
            return
        self._canny_original = None
        self._canny_noisy = None
        self.noisy_image = None
        self._show_images()
