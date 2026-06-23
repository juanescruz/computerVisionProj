"""SUSAN edge & corner detector frame."""
import os
import cv2
import customtkinter as ctk
import numpy as np

from processing.advanced_edge_detection import susan_detector
from processing.noise_contamination import (
    add_gaussian_noise, add_exponential_noise, add_rayleigh_noise,
    add_salt_pepper_noise
)
from gui.utils import convert_cv_to_ctk


def _overlay_marks(img: np.ndarray, mask: np.ndarray, color: tuple) -> np.ndarray:
    """
    Overlay binary mask on image as colored dots.
    color: (R, G, B)
    """
    if img is None:
        return None

    rgb = np.stack([img, img, img], axis=2).astype(np.uint8)

    if mask is not None:
        if mask.shape != img.shape:
            return rgb

        rgb[mask > 0] = color

    return rgb


class SusanFrame(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.noisy_image = None
        self._edges_orig = None
        self._corners_orig = None
        self._edges_noisy = None
        self._corners_noisy = None

        self.title_label = ctk.CTkLabel(
            self, text="Detector SUSAN — Bordes e Esquinas",
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

        self.btn_noise = ctk.CTkButton(
            self.noise_frame, text="Aplicar Ruído",
            command=self.apply_noise, width=110
        )
        self.btn_noise.pack(side="left", padx=8)

        # ── SUSAN controls ──
        self.susan_frame = ctk.CTkFrame(self)
        self.susan_frame.pack(pady=2, fill="x")

        ctk.CTkLabel(self.susan_frame, text="SUSAN t:").pack(side="left", padx=5)
        self.t_entry = ctk.CTkEntry(self.susan_frame, width=50)
        self.t_entry.insert(0, "15")
        self.t_entry.pack(side="left", padx=2)

        self.btn_detect = ctk.CTkButton(
            self.susan_frame, text="Detectar SUSAN",
            command=self.detect_all, width=130
        )
        self.btn_detect.pack(side="left", padx=(15, 5))

        self.btn_reset = ctk.CTkButton(
            self.susan_frame, text="Reset", command=self.reset_all, width=90
        )
        self.btn_reset.pack(side="left", padx=5)

        self.info_label = ctk.CTkLabel(self.susan_frame, text="", text_color="gray")
        self.info_label.pack(side="left", padx=10)

        # ── 2x2 grid display ──
        self.grid_frame = ctk.CTkFrame(self)
        self.grid_frame.pack(fill="both", expand=True, pady=5)
        self.grid_frame.grid_rowconfigure((0, 1, 2), weight=1)
        self.grid_frame.grid_columnconfigure((0, 1), weight=1)

        layout = [
            (0, 0, "original", "Imagen Original"),
            (0, 1, "noisy", "Imagen Contaminada"),

            (1, 0, "edges_orig", "Original — Bordes SUSAN"),
            (1, 1, "edges_noisy", "Contaminada — Bordes SUSAN"),

            (2, 0, "corners_orig", "Original — Esquinas SUSAN"),
            (2, 1, "corners_noisy", "Contaminada — Esquinas SUSAN"),
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
        if selection == "Gaussiano":
            self.noise_p1.delete(0, "end"); self.noise_p1.insert(0, "0")
            self.noise_p2.delete(0, "end"); self.noise_p2.insert(0, "25")
        elif selection == "Exponencial":
            self.noise_p1.delete(0, "end"); self.noise_p1.insert(0, "0.05")
            self.noise_p2.configure(state="disabled")
        elif selection == "Rayleigh":
            self.noise_p1.delete(0, "end"); self.noise_p1.insert(0, "1.2")
            self.noise_p2.configure(state="disabled")
        elif selection == "Sal y Pimienta":
            self.noise_p1.delete(0, "end"); self.noise_p1.insert(0, "0.05")
            self.noise_p2.configure(state="disabled")
        for w in [self.noise_p2]:
            if selection == "Gaussiano":
                w.configure(state="normal")
            else:
                w.configure(state="disabled")

    def _read_noise_params(self):
        ntype = self.noise_var.get()
        pct = self.noise_pct.get() / 100.0
        if ntype == "Gaussiano":
            return ntype, pct, float(self.noise_p1.get()), float(self.noise_p2.get())
        elif ntype == "Exponencial":
            return ntype, pct, float(self.noise_p1.get()), None
        elif ntype == "Rayleigh":
            return ntype, pct, float(self.noise_p1.get()), None
        else:
            return ntype, pct, float(self.noise_p1.get()), None

    def _apply_noise_to(self, img):
        ntype, pct, p1, p2 = self._read_noise_params()
        if ntype == "Gaussiano":
            return add_gaussian_noise(img, pct, p1, p2)
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
        self._edges_noisy = None
        self._corners_noisy = None
        self._show_panels()
        self.info_label.configure(text=f"Ruído {self.noise_var.get()} aplicado", text_color="green")

    # ── SUSAN detection ──
    def _get_params(self):
        try:
            t = float(self.t_entry.get())
            return t
        except ValueError:
            return None

    def detect_all(self):
        if self.app.current_image is None:
            return
        params = self._get_params()
        if params is None:
            self.info_label.configure(text="Parâmetros inválidos", text_color="red")
            return
        t = params

        self._edges_orig, self._corners_orig = susan_detector(
            self.app.current_image, t
        )
        if self.noisy_image is not None:
            self._edges_noisy, self._corners_noisy = susan_detector(
                self.noisy_image, t
            )

        self._show_panels()
        self.info_label.configure(
            text=f"SUSAN t={t:.0f}", text_color="green"
        )
        self._guardar_resultados_susan(t)

    def reset_all(self):
        self.noisy_image = None
        self._edges_orig = None
        self._corners_orig = None
        self._edges_noisy = None
        self._corners_noisy = None
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

    def _make_overlay(self, gray, mask, color):
        if gray is None:
            return None
        return _overlay_marks(gray, mask, color)

    def _show_panels(self):
        gray = self.app.current_image
        gray_noisy = self.noisy_image
        
        self._show_in_panel(gray, "original")
        self._show_in_panel(gray_noisy, "noisy")
        self._show_in_panel(
            self._make_overlay(gray, self._edges_orig, (0, 255, 0)), "edges_orig"
        )
        self._show_in_panel(
            self._make_overlay(gray, self._corners_orig, (255, 0, 0)), "corners_orig"
        )
        self._show_in_panel(
            self._make_overlay(gray_noisy, self._edges_noisy, (0, 255, 0)), "edges_noisy"
        )
        self._show_in_panel(
            self._make_overlay(gray_noisy, self._corners_noisy, (255, 0, 0)), "corners_noisy"
        )

    def _guardar_resultados_susan(self, t):
        base = os.path.join("resultados", "SUSAN")
        os.makedirs(base, exist_ok=True)
        gray_rgb = np.stack([self.app.current_image]*3, axis=2).astype(np.uint8)
        cv2.imwrite(os.path.join(base, "original.png"),
                    cv2.cvtColor(gray_rgb, cv2.COLOR_RGB2BGR))
        if self.noisy_image is not None:
            noisy_rgb = np.stack([self.noisy_image]*3, axis=2).astype(np.uint8)
            cv2.imwrite(os.path.join(base, "ruido.png"),
                        cv2.cvtColor(noisy_rgb, cv2.COLOR_RGB2BGR))
        edges_rgb = _overlay_marks(self.app.current_image, self._edges_orig, (0, 255, 0))
        cv2.imwrite(os.path.join(base, f"bordes_original_t{t:.0f}.png"),
                    cv2.cvtColor(edges_rgb, cv2.COLOR_RGB2BGR))
        corners_rgb = _overlay_marks(self.app.current_image, self._corners_orig, (255, 0, 0))
        cv2.imwrite(os.path.join(base, f"esquinas_original_t{t:.0f}.png"),
                    cv2.cvtColor(corners_rgb, cv2.COLOR_RGB2BGR))
        if self._edges_noisy is not None:
            edges_n_rgb = _overlay_marks(self.noisy_image, self._edges_noisy, (0, 255, 0))
            cv2.imwrite(os.path.join(base, f"bordes_ruido_t{t:.0f}.png"),
                        cv2.cvtColor(edges_n_rgb, cv2.COLOR_RGB2BGR))
        if self._corners_noisy is not None:
            corners_n_rgb = _overlay_marks(self.noisy_image, self._corners_noisy, (255, 0, 0))
            cv2.imwrite(os.path.join(base, f"esquinas_ruido_t{t:.0f}.png"),
                        cv2.cvtColor(corners_n_rgb, cv2.COLOR_RGB2BGR))

    def update_display(self):
        if self.app.current_image is None:
            for k in self._panels:
                self._show_in_panel(None, k)
            self.noisy_image = None
            self._edges_orig = None
            self._corners_orig = None
            self._edges_noisy = None
            self._corners_noisy = None
            return
        self.noisy_image = None
        self._edges_orig = None
        self._corners_orig = None
        self._edges_noisy = None
        self._corners_noisy = None
        self._show_panels()
