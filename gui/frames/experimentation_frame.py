"""Experimentation frame for noise and filter analysis."""
import customtkinter as ctk
import numpy as np
import os
from datetime import datetime
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from processing.noise_contamination import (
    add_gaussian_noise, add_exponential_noise, add_rayleigh_noise,
    add_salt_pepper_noise
)
from processing.spatial_filters import (
    mean_filter, median_filter, weighted_median_filter,
    gaussian_filter, edge_enhancement_filter
)
from gui.utils import convert_cv_to_ctk, plot_histogram


class ExperimentationFrame(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        
        self.title_label = ctk.CTkLabel(self, text="Experimentación", font=("Arial", 16, "bold"))
        self.title_label.pack(pady=5)
        
        self.controls_frame = ctk.CTkFrame(self)
        self.controls_frame.pack(side="left", fill="y", padx=5, pady=5)
        
        self.noise_section_label = ctk.CTkLabel(
            self.controls_frame, text="RUIDO", font=("Arial", 12, "bold")
        )
        self.noise_section_label.pack(pady=5)
        
        self.noise_type_label = ctk.CTkLabel(self.controls_frame, text="Tipo:")
        self.noise_type_label.pack(pady=2)
        self.noise_type_var = ctk.StringVar(value="Sal y Pimienta")
        self.noise_type_menu = ctk.CTkOptionMenu(
            self.controls_frame, variable=self.noise_type_var,
            values=["Gaussiano", "Exponencial", "Rayleigh", "Sal y Pimienta"],
            command=self.on_noise_type_change
        )
        self.noise_type_menu.pack(pady=2)
        
        self.noise_param1_label = ctk.CTkLabel(self.controls_frame, text="Parámetro 1:")
        self.noise_param1_label.pack(pady=2)
        self.noise_param1_entry = ctk.CTkEntry(self.controls_frame, width=120)
        self.noise_param1_entry.insert(0, "0")
        self.noise_param1_entry.pack(pady=2)
        
        self.noise_param2_label = ctk.CTkLabel(self.controls_frame, text="Parámetro 2:")
        self.noise_param2_label.pack(pady=2)
        self.noise_param2_entry = ctk.CTkEntry(self.controls_frame, width=120)
        self.noise_param2_entry.insert(0, "25")
        self.noise_param2_entry.pack(pady=2)
        
        self.noise_param2_label.pack()
        self.noise_param2_entry.pack()
        
        self.noise_percent_label = ctk.CTkLabel(self.controls_frame, text="Porcentaje (%):")
        self.noise_percent_label.pack(pady=2)
        self.noise_percent_slider = ctk.CTkSlider(
            self.controls_frame, from_=0, to=100, number_of_steps=100
        )
        self.noise_percent_slider.set(10)
        self.noise_percent_slider.pack(pady=2)
        
        self.filter_section_label = ctk.CTkLabel(
            self.controls_frame, text="FILTRO", font=("Arial", 12, "bold")
        )
        self.filter_section_label.pack(pady=5)
        
        self.filter_type_label = ctk.CTkLabel(self.controls_frame, text="Tipo:")
        self.filter_type_label.pack(pady=2)
        self.filter_type_var = ctk.StringVar(value="Mediana")
        self.filter_type_menu = ctk.CTkOptionMenu(
            self.controls_frame, variable=self.filter_type_var,
            values=["Media", "Mediana", "Mediana Ponderada", "Gauss", "Realce"],
            command=self.on_filter_type_change
        )
        self.filter_type_menu.pack(pady=2)
        
        self.filter_param1_label = ctk.CTkLabel(self.controls_frame, text="Parámetro 1:")
        self.filter_param1_label.pack(pady=2)
        self.filter_param1_entry = ctk.CTkEntry(self.controls_frame, width=120)
        self.filter_param1_entry.insert(0, "3")
        self.filter_param1_entry.pack(pady=2)
        
        self.filter_param2_label = ctk.CTkLabel(self.controls_frame, text="Parámetro 2:")
        self.filter_param2_label.pack(pady=2)
        self.filter_param2_entry = ctk.CTkEntry(self.controls_frame, width=120)
        self.filter_param2_entry.insert(0, "1.0")
        self.filter_param2_entry.pack(pady=2)
        
        self.filter_param2_label.pack()
        self.filter_param2_entry.pack()
        
        self.btn_apply_sequence = ctk.CTkButton(
            self.controls_frame, text="Aplicar Secuencia Completa",
            command=self.apply_sequence, width=150
        )
        self.btn_apply_sequence.pack(pady=10)
        
        self.btn_export = ctk.CTkButton(
            self.controls_frame, text="Exportar Comparación",
            command=self.export_comparison, width=150
        )
        self.btn_export.pack(pady=5)
        
        self.info_label = ctk.CTkLabel(
            self.controls_frame,
            text="Notas:\n"
            "- Mediana: elimina S&P\n"
            "- Gauss: suaviza ruido gaussiano\n"
            "- Realce: amplifica ruido",
            font=("Arial", 9), text_color="gray"
        )
        self.info_label.pack(pady=10)
        
        self.viz_frame = ctk.CTkFrame(self)
        self.viz_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        
        self.col1_frame = ctk.CTkFrame(self.viz_frame)
        self.col1_frame.pack(side="left", fill="both", expand=True, padx=2)
        self.col2_frame = ctk.CTkFrame(self.viz_frame)
        self.col2_frame.pack(side="left", fill="both", expand=True, padx=2)
        self.col3_frame = ctk.CTkFrame(self.viz_frame)
        self.col3_frame.pack(side="left", fill="both", expand=True, padx=2)
        
        ctk.CTkLabel(self.col1_frame, text="Original", font=("Arial", 11, "bold")).pack(pady=2)
        self.orig_label = ctk.CTkLabel(self.col1_frame, text="Sin imagen", text_color="gray")
        self.orig_label.pack(pady=2)
        self.hist1_frame = ctk.CTkFrame(self.col1_frame)
        self.hist1_frame.pack(fill="both", expand=True, pady=2)
        
        ctk.CTkLabel(self.col2_frame, text="Con Ruido", font=("Arial", 11, "bold")).pack(pady=2)
        self.noisy_label = ctk.CTkLabel(self.col2_frame, text="Sin imagen", text_color="gray")
        self.noisy_label.pack(pady=2)
        self.hist2_frame = ctk.CTkFrame(self.col2_frame)
        self.hist2_frame.pack(fill="both", expand=True, pady=2)
        
        ctk.CTkLabel(self.col3_frame, text="Filtrada", font=("Arial", 11, "bold")).pack(pady=2)
        self.filtered_label = ctk.CTkLabel(self.col3_frame, text="Sin imagen", text_color="gray")
        self.filtered_label.pack(pady=2)
        self.hist3_frame = ctk.CTkFrame(self.col3_frame)
        self.hist3_frame.pack(fill="both", expand=True, pady=2)
        
        self.img_original = None
        self.img_noisy = None
        self.img_filtered = None
        self.hist1_canvas = None
        self.hist2_canvas = None
        self.hist3_canvas = None
    
    def on_noise_type_change(self, selection):
        if selection == "Gaussiano":
            self.noise_param1_label.configure(text="Media (μ):")
            self.noise_param1_entry.delete(0, "end")
            self.noise_param1_entry.insert(0, "0")
            self.noise_param2_label.configure(text="Sigma (σ):")
            self.noise_param2_entry.delete(0, "end")
            self.noise_param2_entry.insert(0, "25")
            self.noise_param2_label.pack()
            self.noise_param2_entry.pack()
        elif selection == "Exponencial":
            self.noise_param1_label.configure(text="Lambda (λ):")
            self.noise_param1_entry.delete(0, "end")
            self.noise_param1_entry.insert(0, "0.05")
            self.noise_param2_label.pack_remove()
            self.noise_param2_entry.pack_remove()
        elif selection == "Rayleigh":
            self.noise_param1_label.configure(text="Xi (ξ):")
            self.noise_param1_entry.delete(0, "end")
            self.noise_param1_entry.insert(0, "1.2")
            self.noise_param2_label.pack_remove()
            self.noise_param2_entry.pack_remove()
        elif selection == "Sal y Pimienta":
            self.noise_param1_label.configure(text="p:")
            self.noise_param1_entry.delete(0, "end")
            self.noise_param1_entry.insert(0, "0.1")
            self.noise_param2_label.pack_remove()
            self.noise_param2_entry.pack_remove()
    
    def on_filter_type_change(self, selection):
        if selection in ["Media", "Mediana"]:
            self.filter_param1_label.configure(text="Tamaño kernel:")
            self.filter_param1_entry.delete(0, "end")
            self.filter_param1_entry.insert(0, "3")
            self.filter_param1_label.pack()
            self.filter_param1_entry.pack()
            self.filter_param2_label.pack_remove()
            self.filter_param2_entry.pack_remove()
        elif selection == "Mediana Ponderada":
            self.filter_param1_label.configure(text="Pesos:")
            self.filter_param1_entry.delete(0, "end")
            self.filter_param1_entry.insert(0, "1,1,1;1,3,1;1,1,1")
            self.filter_param1_label.pack()
            self.filter_param1_entry.pack()
            self.filter_param2_label.pack_remove()
            self.filter_param2_entry.pack_remove()
        elif selection == "Gauss":
            self.filter_param1_label.pack_remove()
            self.filter_param1_entry.pack_remove()
            self.filter_param2_label.configure(text="Sigma (σ):")
            self.filter_param2_entry.delete(0, "end")
            self.filter_param2_entry.insert(0, "1.0")
            self.filter_param2_label.pack()
            self.filter_param2_entry.pack()
        elif selection == "Realce":
            self.filter_param1_label.pack_remove()
            self.filter_param1_entry.pack_remove()
            self.filter_param2_label.pack_remove()
            self.filter_param2_entry.pack_remove()
    
    def apply_sequence(self):
        if self.app.current_image is None:
            self._show_warning()
            return
        
        self.img_original = self.app.current_image
        percentage = self.noise_percent_slider.get() / 100.0
        noise_type = self.noise_type_var.get()
        
        if noise_type == "Gaussiano":
            try:
                mean = float(self.noise_param1_entry.get())
                sigma = float(self.noise_param2_entry.get())
            except:
                mean, sigma = 0.0, 25.0
            self.img_noisy = add_gaussian_noise(self.img_original, percentage, mean, sigma)
        elif noise_type == "Exponencial":
            try:
                lam = float(self.noise_param1_entry.get())
            except:
                lam = 0.05
            self.img_noisy = add_exponential_noise(self.img_original, percentage, lam)
        elif noise_type == "Rayleigh":
            try:
                xi = float(self.noise_param1_entry.get())
            except:
                xi = 1.2
            self.img_noisy = add_rayleigh_noise(self.img_original, percentage, xi)
        elif noise_type == "Sal y Pimienta":
            try:
                p = float(self.noise_param1_entry.get())
            except:
                p = 0.1
            self.img_noisy = add_salt_pepper_noise(self.img_original, p)
        
        filter_type = self.filter_type_var.get()
        
        if filter_type == "Media":
            try:
                k = int(self.filter_param1_entry.get())
            except:
                k = 3
            self.img_filtered = mean_filter(self.img_noisy, k)
        elif filter_type == "Mediana":
            try:
                k = int(self.filter_param1_entry.get())
            except:
                k = 3
            self.img_filtered = median_filter(self.img_noisy, k)
        elif filter_type == "Mediana Ponderada":
            weights_str = self.filter_param1_entry.get()
            weights = self._parse_weights(weights_str)
            if weights is None:
                weights = np.array([[1,1,1],[1,3,1],[1,1,1]], dtype=np.int32)
            self.img_filtered = weighted_median_filter(self.img_noisy, weights)
        elif filter_type == "Gauss":
            try:
                sigma = float(self.filter_param2_entry.get())
            except:
                sigma = 1.0
            self.img_filtered = gaussian_filter(self.img_noisy, sigma)
        elif filter_type == "Realce":
            self.img_filtered = edge_enhancement_filter(self.img_noisy)
        
        self.update_display()
    
    def _parse_weights(self, weights_str):
        try:
            rows = weights_str.split(';')
            weights = []
            for row in rows:
                values = [int(x.strip()) for x in row.split(',')]
                weights.append(values)
            return np.array(weights, dtype=np.int32)
        except:
            return None
    
    def update_display(self):
        for canvas, frame, img in [
            (self.hist1_canvas, self.hist1_frame, self.img_original),
            (self.hist2_canvas, self.hist2_frame, self.img_noisy),
            (self.hist3_canvas, self.hist3_frame, self.img_filtered)
        ]:
            if canvas:
                canvas.get_tk_widget().destroy()
        
        if self.img_original is not None:
            self._show_image_label(self.orig_label, self.img_original)
            self.hist1_canvas = plot_histogram(self.img_original, self.hist1_frame)
            self.hist1_canvas.get_tk_widget().pack(fill="both", expand=True)
        
        if self.img_noisy is not None:
            self._show_image_label(self.noisy_label, self.img_noisy)
            self.hist2_canvas = plot_histogram(self.img_noisy, self.hist2_frame)
            self.hist2_canvas.get_tk_widget().pack(fill="both", expand=True)
        
        if self.img_filtered is not None:
            self._show_image_label(self.filtered_label, self.img_filtered)
            self.hist3_canvas = plot_histogram(self.img_filtered, self.hist3_frame)
            self.hist3_canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def _show_image_label(self, label, img):
        h, w = img.shape
        ctk_img = convert_cv_to_ctk(img, size=(min(w, 200), min(h, 180)))
        label.configure(image=ctk_img, text="")
        label.image = ctk_img
    
    def _show_warning(self):
        import tkinter.messagebox as msgbox
        msgbox.showwarning("Advertencia", "Cargar una imagen primero desde Inicio.")
    
    def export_comparison(self):
        if self.img_original is None or self.img_noisy is None or self.img_filtered is None:
            import tkinter.messagebox as msgbox
            msgbox.showwarning("Advertencia", "Aplicar la secuencia primero.")
            return
        
        export_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "exports")
        os.makedirs(export_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(export_dir, f"comparacion_{timestamp}.png")
        
        fig = Figure(figsize=(12, 8), dpi=100)
        
        for i, (img, title) in enumerate([
            (self.img_original, "Original"),
            (self.img_noisy, "Con Ruido"),
            (self.img_filtered, "Filtrada")
        ], 1):
            ax = fig.add_subplot(2, 3, i)
            ax.imshow(img, cmap="gray")
            ax.set_title(title)
            ax.axis("off")
        
        for i, (img, title) in enumerate([
            (self.img_original, "Hist Original"),
            (self.img_noisy, "Hist Ruido"),
            (self.img_filtered, "Hist Filtrada")
        ], 4):
            from processing.point_operators import calc_histogram
            hist = calc_histogram(img)
            ax = fig.add_subplot(2, 3, i)
            ax.bar(range(256), hist, width=1.0, color="gray")
            ax.set_xlim(0, 255)
            ax.set_title(title)
        
        fig.tight_layout()
        fig.savefig(filepath, dpi=100)
        
        import tkinter.messagebox as msgbox
        msgbox.showinfo("Exportado", f"Figura guardada en:\n{filepath}")