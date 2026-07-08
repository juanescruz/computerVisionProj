"""SIFT feature matching frame with transformations and 2-image support."""
import customtkinter as ctk
import numpy as np
from tkinter import filedialog

from processing.feature_matching import (
    detect_sift, match_features, draw_keypoints, draw_matches,
    apply_rotation, apply_scale, apply_translation,
    apply_illumination, apply_perspective, apply_gaussian_noise
)
from gui.utils import convert_cv_to_ctk


class SiftFrame(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.query_image = None
        self._loaded_query = None
        self._loaded_query_path = None
        self._kp_ref = None
        self._desc_ref = None
        self._kp_query = None
        self._desc_query = None
        self._matches = None
        self._match_vis = None
        self._transform_applied = "nenhuma"

        self.scroll = ctk.CTkScrollableFrame(self)
        self.scroll.pack(fill="both", expand=True)

        self.title_label = ctk.CTkLabel(
            self.scroll, text="SIFT — Reconocimiento de Objetos",
            font=("Arial", 16, "bold")
        )
        self.title_label.pack(pady=5)

        # ── Image source controls ──
        self.src_frame = ctk.CTkFrame(self.scroll)
        self.src_frame.pack(pady=2, fill="x")

        self.btn_load2 = ctk.CTkButton(
            self.src_frame, text="Cargar 2ª Imagem",
            command=self.load_second_image, width=160
        )
        self.btn_load2.pack(side="left", padx=5)

        self.src_label = ctk.CTkLabel(
            self.src_frame, text="Modo: 1 imagen (referencia + transformación)",
            text_color="gray"
        )
        self.src_label.pack(side="left", padx=10)

        self.apply_tf_var = ctk.BooleanVar(value=True)
        self.apply_tf_cb = ctk.CTkCheckBox(
            self.src_frame, text="Aplicar transformación",
            variable=self.apply_tf_var, width=140
        )
        self.apply_tf_cb.pack(side="left", padx=5)

        # ── Transformation controls ──
        self.tf_frame = ctk.CTkFrame(self.scroll)
        self.tf_frame.pack(pady=2, fill="x")

        self.tf_var = ctk.StringVar(value="Rotación")
        self.tf_menu = ctk.CTkOptionMenu(
            self.tf_frame, variable=self.tf_var,
            values=["Rotación", "Escala", "Traslación",
                    "Iluminación", "Perspectiva", "Ruído Gaussiano"],
            command=self.on_tf_change, width=130
        )
        self.tf_menu.pack(side="left", padx=5)

        self.param1_label = ctk.CTkLabel(self.tf_frame, text="Ángulo:")
        self.param1_label.pack(side="left", padx=(8, 2))
        self.param1_entry = ctk.CTkEntry(self.tf_frame, width=55)
        self.param1_entry.insert(0, "45")
        self.param1_entry.pack(side="left", padx=2)

        self.param2_label = ctk.CTkLabel(self.tf_frame, text="")
        self.param2_label.pack(side="left", padx=(8, 2))
        self.param2_entry = ctk.CTkEntry(self.tf_frame, width=55)
        self.param2_entry.pack(side="left", padx=2)

        self.noise_both_var = ctk.BooleanVar(value=False)
        self.noise_both_cb = ctk.CTkCheckBox(
            self.tf_frame, text="Ruído em ambas",
            variable=self.noise_both_var, width=100
        )
        self.noise_both_cb.pack(side="left", padx=5)

        self.btn_apply = ctk.CTkButton(
            self.tf_frame, text="Aplicar + Matching",
            command=self.apply_and_match, width=150
        )
        self.btn_apply.pack(side="left", padx=(8, 5))

        self.btn_reset = ctk.CTkButton(
            self.tf_frame, text="Reset", command=self.reset_all, width=80
        )
        self.btn_reset.pack(side="left", padx=5)

        self.info_label = ctk.CTkLabel(self.tf_frame, text="", text_color="gray")
        self.info_label.pack(side="left", padx=5)

        # ── Stats panel ──
        self.stats_frame = ctk.CTkFrame(self.scroll)
        self.stats_frame.pack(pady=2, fill="x")
        self.stats_label = ctk.CTkLabel(
            self.stats_frame, text="", font=("Arial", 12)
        )
        self.stats_label.pack()

        # ── Match visualization (first, large) ──
        self.match_frame = ctk.CTkFrame(self.scroll)
        self.match_frame.pack(fill="both", expand=True, pady=2)
        ctk.CTkLabel(
            self.match_frame, text="Correspondencias (Matches)",
            font=("Arial", 11, "bold")
        ).pack(pady=2)
        self.match_label = ctk.CTkLabel(
            self.match_frame, text="Sin correspondencias", text_color="gray"
        )
        self.match_label.pack(pady=5, fill="both", expand=True)

        # ── Display grid (below matches) ──
        self.grid_frame = ctk.CTkFrame(self.scroll)
        self.grid_frame.pack(fill="x", pady=5)
        self.grid_frame.grid_rowconfigure((0, 1), weight=1)
        self.grid_frame.grid_columnconfigure((0, 1), weight=1)

        layout = [
            (0, 0, "ref", "Referencia (Original)"),
            (0, 1, "query", "Consulta"),
            (1, 0, "kp_ref", "Keypoints — Referencia"),
            (1, 1, "kp_query", "Keypoints — Consulta"),
        ]
        self._panels = {}
        for r, c, key, title in layout:
            panel = ctk.CTkFrame(self.grid_frame)
            panel.grid(row=r, column=c, sticky="nsew", padx=3, pady=3)
            panel.grid_rowconfigure(0, weight=1)
            panel.grid_columnconfigure(0, weight=1)

            lbl = ctk.CTkLabel(panel, text="Sin imagen", text_color="gray")
            lbl.grid(row=0, column=0, pady=3)
            ctk.CTkLabel(panel, text=title, font=("Arial", 11, "bold")).grid(row=1, column=0, pady=1)

            self._panels[key] = {"label": lbl}

    def load_second_image(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.bmp *.tif *.tiff"), ("All files", "*.*")]
        )
        if not filepath:
            return
        import cv2
        img = cv2.imread(filepath, cv2.IMREAD_COLOR)
        if img is None:
            self.info_label.configure(text="Error al cargar imagen", text_color="red")
            return
        self._loaded_query = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._loaded_query_path = filepath
        fname = filepath.replace("\\", "/").split("/")[-1]
        self.src_label.configure(
            text=f"2 imagens: ref + {fname}",
            text_color="green"
        )
        self.apply_tf_var.set(False)
        self.info_label.configure(text="2ª imagen cargada", text_color="green")

    def on_tf_change(self, selection):
        self.param2_entry.configure(state="normal")
        self.noise_both_cb.configure(state="normal")
        if selection == "Rotación":
            self.param1_label.configure(text="Ángulo:")
            self.param1_entry.delete(0, "end"); self.param1_entry.insert(0, "45")
            self.param2_label.configure(text="")
            self.param2_entry.configure(state="disabled")
            self.noise_both_cb.configure(state="disabled")
        elif selection == "Escala":
            self.param1_label.configure(text="Fator:")
            self.param1_entry.delete(0, "end"); self.param1_entry.insert(0, "0.5")
            self.param2_label.configure(text="")
            self.param2_entry.configure(state="disabled")
            self.noise_both_cb.configure(state="disabled")
        elif selection == "Traslación":
            self.param1_label.configure(text="dx:")
            self.param1_entry.delete(0, "end"); self.param1_entry.insert(0, "20")
            self.param2_label.configure(text="dy:")
            self.param2_entry.delete(0, "end"); self.param2_entry.insert(0, "15")
            self.noise_both_cb.configure(state="disabled")
        elif selection == "Iluminación":
            self.param1_label.configure(text="Alpha:")
            self.param1_entry.delete(0, "end"); self.param1_entry.insert(0, "0.6")
            self.param2_label.configure(text="Beta:")
            self.param2_entry.delete(0, "end"); self.param2_entry.insert(0, "30")
            self.noise_both_cb.configure(state="disabled")
        elif selection == "Perspectiva":
            self.param1_label.configure(text="Skew:")
            self.param1_entry.delete(0, "end"); self.param1_entry.insert(0, "0.2")
            self.param2_label.configure(text="")
            self.param2_entry.configure(state="disabled")
            self.noise_both_cb.configure(state="disabled")
        elif selection == "Ruído Gaussiano":
            self.param1_label.configure(text="σ:")
            self.param1_entry.delete(0, "end"); self.param1_entry.insert(0, "30")
            self.param2_label.configure(text="")
            self.param2_entry.configure(state="disabled")

    def _get_ref(self):
        return (self.app.current_image_color if self.app.current_image_color is not None
                else self.app.current_image)

    def _build_query(self, img):
        if not self.apply_tf_var.get():
            return img.copy()
        self._transform_applied = self.tf_var.get()
        tf = self._transform_applied
        p1 = float(self.param1_entry.get()) if self.param1_entry.get() else 0
        p2 = float(self.param2_entry.get()) if self.param2_entry.get() else 0
        if tf == "Rotación":
            return apply_rotation(img, p1)
        elif tf == "Escala":
            return apply_scale(img, p1)
        elif tf == "Traslación":
            return apply_translation(img, int(p1), int(p2))
        elif tf == "Iluminación":
            return apply_illumination(img, p1, int(p2))
        elif tf == "Perspectiva":
            return apply_perspective(img, p1)
        elif tf == "Ruído Gaussiano":
            return apply_gaussian_noise(img, 0, p1)
        return img.copy()

    def apply_and_match(self):
        if self.app.current_image is None:
            return

        ref = self._get_ref()

        # Determine query source
        if self._loaded_query is not None:
            query_src = self._loaded_query
        else:
            query_src = ref

        self.query_image = self._build_query(query_src)
        self._transform_applied = self.tf_var.get() if self.apply_tf_var.get() else "nenhuma"

        # Optional noise on both
        if self.noise_both_var.get():
            sigma = float(self.param1_entry.get()) if self.param1_entry.get() else 30
            ref = apply_gaussian_noise(ref, 0, sigma)
            self.query_image = apply_gaussian_noise(self.query_image, 0, sigma)

        # SIFT detection
        self._kp_ref, self._desc_ref = detect_sift(ref)
        self._kp_query, self._desc_query = detect_sift(self.query_image)

        # Matching
        self._matches, all_m = match_features(self._desc_ref, self._desc_query)

        # Stats
        n_ref = len(self._kp_ref) if self._kp_ref is not None else 0
        n_query = len(self._kp_query) if self._kp_query is not None else 0
        n_matches = len(self._matches) if self._matches is not None else 0
        n_all = len(all_m) if all_m is not None else 0
        ratio = n_matches / min(n_ref, n_query) * 100 if min(n_ref, n_query) > 0 else 0

        self.stats_label.configure(
            text=f"Transformación: {self._transform_applied}  |  "
                 f"Keypoints ref: {n_ref}  query: {n_query}  |  "
                 f"Matches: {n_matches}/{n_all}  ({ratio:.1f}%)"
        )

        # Visualizations
        kp_ref_rgb = draw_keypoints(ref, self._kp_ref)
        kp_query_rgb = draw_keypoints(self.query_image, self._kp_query)

        # Match visualization
        if self._matches and len(self._matches) > 0:
            self._match_vis = draw_matches(
                ref, self._kp_ref, self.query_image, self._kp_query,
                self._matches
            )
            h, w = self._match_vis.shape[:2]
            scale = min(1200 / w, 900 / h, 1.0)
            disp_w = int(w * scale)
            disp_h = int(h * scale)
            pil_img = self._pil_from_cv(self._match_vis, (disp_w, disp_h))
            self.match_label.configure(image=pil_img, text="")
            self.match_label.image = pil_img
        else:
            self.match_label.configure(image=None, text="Sin correspondencias", text_color="gray")

        # Show panels
        ref_rgb = np.stack([ref]*3, axis=2).astype(np.uint8) if ref.ndim == 2 else ref.copy()
        query_rgb = (np.stack([self.query_image]*3, axis=2).astype(np.uint8)
                     if self.query_image.ndim == 2 else self.query_image.copy())

        self._show_in_panel(ref_rgb, "ref")
        self._show_in_panel(query_rgb, "query")
        self._show_in_panel(kp_ref_rgb, "kp_ref")
        self._show_in_panel(kp_query_rgb, "kp_query")

        self.info_label.configure(
            text=f"SIFT: {n_matches} correspondencias validas {n_all}",
            text_color="green"
        )

    def _pil_from_cv(self, img_rgb, size=None):
        from PIL import Image as PILImage
        if size is None:
            h, w = img_rgb.shape[:2]
            disp_w = min(w, 500)
            disp_h = int(h * disp_w / w)
            size = (disp_w, disp_h)
        pil = PILImage.fromarray(img_rgb)
        return ctk.CTkImage(light_image=pil, dark_image=pil, size=size)

    def _show_in_panel(self, img_rgb, key):
        panel = self._panels[key]
        if img_rgb is None:
            panel["label"].configure(image=None, text="Sin resultado", text_color="gray")
            return
        h, w = img_rgb.shape[:2]
        ctk_img = convert_cv_to_ctk(img_rgb, size=(min(w, 280), min(h, 220)))
        panel["label"].configure(image=ctk_img, text="")
        panel["label"].image = ctk_img

    def reset_all(self):
        self.query_image = None
        self._kp_ref = None
        self._desc_ref = None
        self._kp_query = None
        self._desc_query = None
        self._matches = None
        self._match_vis = None
        self.match_label.configure(image=None, text="Sin correspondencias", text_color="gray")
        self.stats_label.configure(text="")
        self.info_label.configure(text="")
        for k in self._panels:
            self._panels[k]["label"].configure(image=None, text="Sin imagen", text_color="gray")

    def update_display(self):
        if self.app.current_image is None:
            self.reset_all()
            return
        self.reset_all()
