"""Hough Transform frame — line detection with noise comparison."""
import customtkinter as ctk
import numpy as np

from processing.hough_transform import hough_lines, draw_hough_lines
from processing.noise_contamination import (
    add_gaussian_noise, add_exponential_noise, add_rayleigh_noise,
    add_salt_pepper_noise
)
from gui.utils import convert_cv_to_ctk


class HoughFrame(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.noisy_image = None
        self._hough_orig = None
        self._hough_noisy = None

        self.title_label = ctk.CTkLabel(
            self, text="Transformada de Hough — Detecção de Retas",
            font=("Arial", 16, "bold")
        )
        self.title_label.pack(pady=5)

        # ── Noise controls ──
        self.noise_frame = ctk.CTkFrame(self)
        self.noise_frame.pack(pady=2, fill="x")

        ctk.CTkLabel(self.noise_frame, text="Ruído:").pack(side="left", padx=5)
        self.noise_var = ctk.StringVar(value="Gaussiano")
        self.noise_menu = ctk.CTkOptionMenu(
            self.noise_frame, variable=self.noise_var,
            values=["Gaussiano", "Exponencial", "Rayleigh", "Sal y Pimienta"],
            command=self.on_noise_change, width=110
        )
        self.noise_menu.pack(side="left", padx=2)

        ctk.CTkLabel(self.noise_frame, text="μ/λ/ξ:").pack(side="left", padx=(8, 2))
        self.noise_p1 = ctk.CTkEntry(self.noise_frame, width=55)
        self.noise_p1.insert(0, "0")
        self.noise_p1.pack(side="left", padx=2)

        ctk.CTkLabel(self.noise_frame, text="σ:").pack(side="left", padx=(8, 2))
        self.noise_p2 = ctk.CTkEntry(self.noise_frame, width=55)
        self.noise_p2.insert(0, "25")
        self.noise_p2.pack(side="left", padx=2)

        ctk.CTkLabel(self.noise_frame, text="%:").pack(side="left", padx=(8, 2))
        self.noise_pct = ctk.CTkSlider(
            self.noise_frame, from_=0, to=100, number_of_steps=100, width=90
        )
        self.noise_pct.set(10)
        self.noise_pct.pack(side="left", padx=2)
        self.noise_pct_label = ctk.CTkLabel(self.noise_frame, text="10%")
        self.noise_pct_label.pack(side="left", padx=2)

        self.btn_noise = ctk.CTkButton(
            self.noise_frame, text="Aplicar Ruído",
            command=self.apply_noise, width=110
        )
        self.btn_noise.pack(side="left", padx=8)

        # ── Hough controls ──
        self.hough_frame = ctk.CTkFrame(self)
        self.hough_frame.pack(pady=2, fill="x")

        # Canny params
        ctk.CTkLabel(self.hough_frame, text="Canny σ:").pack(side="left", padx=5)
        self.canny_sigma = ctk.CTkEntry(self.hough_frame, width=50)
        self.canny_sigma.insert(0, "1.0")
        self.canny_sigma.pack(side="left", padx=2)

        ctk.CTkLabel(self.hough_frame, text="Low:").pack(side="left", padx=(5, 2))
        self.canny_low = ctk.CTkEntry(self.hough_frame, width=45)
        self.canny_low.insert(0, "50")
        self.canny_low.pack(side="left", padx=2)

        ctk.CTkLabel(self.hough_frame, text="High:").pack(side="left", padx=(5, 2))
        self.canny_high = ctk.CTkEntry(self.hough_frame, width=45)
        self.canny_high.insert(0, "150")
        self.canny_high.pack(side="left", padx=2)

        # Hough params
        ctk.CTkLabel(self.hough_frame, text="Vote threshold:").pack(side="left", padx=(5, 2))
        self.vote_th = ctk.CTkEntry(self.hough_frame, width=40)
        self.vote_th.insert(0, "70")
        self.vote_th.pack(side="left", padx=2)

        self.btn_detect = ctk.CTkButton(
            self.hough_frame, text="Detectar Retas",
            command=self.detect_all, width=120
        )
        self.btn_detect.pack(side="left", padx=(10, 5))

        self.btn_reset = ctk.CTkButton(
            self.hough_frame, text="Reset", command=self.reset_all, width=80
        )
        self.btn_reset.pack(side="left", padx=5)

        self.info_label = ctk.CTkLabel(self.hough_frame, text="", text_color="gray")
        self.info_label.pack(side="left", padx=5)

        # ── 2x2 grid display ──
        self.grid_frame = ctk.CTkFrame(self)
        self.grid_frame.pack(fill="both", expand=True, pady=5)
        self.grid_frame.grid_rowconfigure((0, 1), weight=1)
        self.grid_frame.grid_columnconfigure((0, 1), weight=1)

        layout = [
            (0, 0, "orig", "Original"),
            (0, 1, "noisy", "Com Ruído"),
            (1, 0, "hough_orig", "Hough — Original"),
            (1, 1, "hough_noisy", "Hough — Com Ruído"),
        ]
        self._panels = {}
        for r, c, key, title in layout:
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
        state_norm = "normal" if selection == "Gaussiano" else "disabled"
        self.noise_p2.configure(state=state_norm)
        if selection == "Gaussiano":
            self.noise_p1.delete(0, "end"); self.noise_p1.insert(0, "0")
            self.noise_p2.delete(0, "end"); self.noise_p2.insert(0, "25")
        elif selection == "Exponencial":
            self.noise_p1.delete(0, "end"); self.noise_p1.insert(0, "0.05")
        elif selection == "Rayleigh":
            self.noise_p1.delete(0, "end"); self.noise_p1.insert(0, "1.2")
        elif selection == "Sal y Pimienta":
            self.noise_p1.delete(0, "end"); self.noise_p1.insert(0, "0.05")

    def _apply_noise_to(self, img):
        ntype = self.noise_var.get()
        pct = self.noise_pct.get() / 100.0
        p1 = float(self.noise_p1.get())
        if ntype == "Gaussiano":
            return add_gaussian_noise(img, pct, p1, float(self.noise_p2.get()))
        elif ntype == "Exponencial":
            return add_exponential_noise(img, pct, p1)
        elif ntype == "Rayleigh":
            return add_rayleigh_noise(img, pct, p1)
        else:
            return add_salt_pepper_noise(img, p1)

    def apply_noise(self):
        if self.app.current_image is None:
            return
        try:
            self.noisy_image = self._apply_noise_to(self.app.current_image)
        except Exception as e:
            self.info_label.configure(text=f"Erro: {e}", text_color="red")
            return
        self._hough_noisy = None
        self._show_panels()
        self.info_label.configure(text=f"Ruído {self.noise_var.get()} aplicado", text_color="green")

    # ── Hough detection ──
    def _get_hough(self, img):
        sigma = float(self.canny_sigma.get())
        low = float(self.canny_low.get())
        high = float(self.canny_high.get())
        v_th = self.vote_th.get()
        threshold = int(v_th) if v_th.strip() else None

        edges, acc, thetas, rhos, lines = hough_lines(
            img, threshold=threshold,
            edge_sigma=sigma, edge_low=low, edge_high=high
        )
        return lines

    def detect_all(self):
        if self.app.current_image is None:
            return
        try:
            lines_orig = self._get_hough(self.app.current_image)
            self._hough_orig = draw_hough_lines(self.app.current_image, lines_orig)
            if self.noisy_image is not None:
                lines_noisy = self._get_hough(self.noisy_image)
                self._hough_noisy = draw_hough_lines(self.noisy_image, lines_noisy)
            info = f"{len(lines_orig)} retas na original"
            if self.noisy_image is not None:
                info += f", {len(lines_noisy)} no ruído"
            self.info_label.configure(text=info, text_color="green")
        except Exception as e:
            self.info_label.configure(text=f"Erro: {e}", text_color="red")
            return
        self._show_panels()

    def reset_all(self):
        self.noisy_image = None
        self._hough_orig = None
        self._hough_noisy = None
        self.info_label.configure(text="")
        self._show_panels()

    # ── Display ──
    def _show_in_panel(self, img_rgb, key):
        panel = self._panels[key]
        if img_rgb is None:
            panel["label"].configure(image=None, text="Sem resultado", text_color="gray")
            return
        h, w = img_rgb.shape[:2]
        ctk_img = convert_cv_to_ctk(img_rgb, size=(min(w, 280), min(h, 220)))
        panel["label"].configure(image=ctk_img, text="")
        panel["label"].image = ctk_img

    def _show_panels(self):
        gray = self.app.current_image
        self._show_in_panel(gray if gray is None else np.stack([gray, gray, gray], axis=2).astype(np.uint8) if gray.ndim == 2 else gray, "orig")
        if self.noisy_image is not None:
            noisy_rgb = np.stack([self.noisy_image]*3, axis=2).astype(np.uint8) if self.noisy_image.ndim == 2 else self.noisy_image
            self._show_in_panel(noisy_rgb, "noisy")
        else:
            self._show_in_panel(None, "noisy")
        self._show_in_panel(self._hough_orig, "hough_orig")
        self._show_in_panel(self._hough_noisy, "hough_noisy")

    def update_display(self):
        if self.app.current_image is None:
            for k in self._panels:
                self._show_in_panel(None, k)
            self.noisy_image = None
            self._hough_orig = None
            self._hough_noisy = None
            return
        self.noisy_image = None
        self._hough_orig = None
        self._hough_noisy = None
        self._show_panels()
