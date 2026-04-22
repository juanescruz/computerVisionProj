"""Spatial filters frame."""
import customtkinter as ctk
import numpy as np

from processing.spatial_filters import (
    mean_filter, median_filter, weighted_median_filter,
    gaussian_filter, edge_enhancement_filter
)
from gui.utils import convert_cv_to_ctk, plot_histogram


class SpatialFiltersFrame(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        
        self.title_label = ctk.CTkLabel(self, text="Filtros Espaciales", font=("Arial", 16, "bold"))
        self.title_label.pack(pady=5)
        
        self.controls_frame = ctk.CTkFrame(self)
        self.controls_frame.pack(pady=5)
        
        self.filter_label = ctk.CTkLabel(self.controls_frame, text="Tipo de filtro:")
        self.filter_label.grid(row=0, column=0, padx=5, pady=5)
        
        self.filter_var = ctk.StringVar(value="Media")
        self.filter_menu = ctk.CTkOptionMenu(
            self.controls_frame, variable=self.filter_var,
            values=["Media", "Mediana", "Mediana Ponderada", "Gauss", "Realce"],
            command=self.on_filter_change
        )
        self.filter_menu.grid(row=0, column=1, padx=5, pady=5)
        
        self.param1_label = ctk.CTkLabel(self.controls_frame, text="Tamaño kernel:")
        self.param1_label.grid(row=1, column=0, padx=5, pady=5)
        self.param1_entry = ctk.CTkEntry(self.controls_frame, width=150)
        self.param1_entry.insert(0, "3")
        self.param1_entry.grid(row=1, column=1, padx=5, pady=5)
        
        self.param2_label = ctk.CTkLabel(self.controls_frame, text="Sigma (σ):")
        self.param2_label.grid(row=2, column=0, padx=5, pady=5)
        self.param2_entry = ctk.CTkEntry(self.controls_frame, width=150)
        self.param2_entry.insert(0, "1.0")
        self.param2_entry.grid(row=2, column=1, padx=5, pady=5)
        
        self.param2_label.grid_remove()
        self.param2_entry.grid_remove()
        
        self.btn_apply = ctk.CTkButton(
            self.controls_frame, text="Aplicar Filtro",
            command=self.apply_filter, width=150
        )
        self.btn_apply.grid(row=3, column=0, columnspan=2, pady=10)
        
        self.comparison_frame = ctk.CTkFrame(self)
        self.comparison_frame.pack(fill="both", expand=True, pady=5)
        
        self.left_frame = ctk.CTkFrame(self.comparison_frame)
        self.left_frame.pack(side="left", fill="both", expand=True, padx=5)
        
        self.right_frame = ctk.CTkFrame(self.comparison_frame)
        self.right_frame.pack(side="right", fill="both", expand=True, padx=5)
        
        self.orig_label = ctk.CTkLabel(self.left_frame, text="Sin imagen", text_color="gray")
        self.orig_label.pack(pady=5)
        ctk.CTkLabel(self.left_frame, text="Original", font=("Arial", 12)).pack(pady=2)
        
        self.proc_label = ctk.CTkLabel(self.right_frame, text="Sin resultado", text_color="gray")
        self.proc_label.pack(pady=5)
        ctk.CTkLabel(self.right_frame, text="Filtrada", font=("Arial", 12)).pack(pady=2)
        
        self.hist_orig_frame = ctk.CTkFrame(self.left_frame)
        self.hist_orig_frame.pack(fill="both", expand=True, pady=5)
        
        self.hist_proc_frame = ctk.CTkFrame(self.right_frame)
        self.hist_proc_frame.pack(fill="both", expand=True, pady=5)
        
        self.hist_orig_canvas = None
        self.hist_proc_canvas = None
    
    def on_filter_change(self, selection):
        if selection in ["Media", "Mediana"]:
            self.param1_label.configure(text="Tamaño kernel:")
            self.param1_entry.delete(0, "end")
            self.param1_entry.insert(0, "3")
            self.param1_label.grid()
            self.param1_entry.grid()
            self.param2_label.grid_remove()
            self.param2_entry.grid_remove()
        elif selection == "Mediana Ponderada":
            self.param1_label.configure(text="Dimensión (n):")
            self.param1_entry.delete(0, "end")
            self.param1_entry.insert(0, "3")
            self.param1_label.grid()
            self.param1_entry.grid()
            self.param2_label.configure(text="Pesos (fila1;fila2;...):")
            self.param2_entry.delete(0, "end")
            self.param2_entry.insert(0, "1,1,1; 1,3,1; 1,1,1")
            self.param2_label.grid()
            self.param2_entry.grid()
        elif selection == "Gauss":
            self.param1_label.grid_remove()
            self.param1_entry.grid_remove()
            self.param2_label.configure(text="Sigma (σ):")
            self.param2_entry.delete(0, "end")
            self.param2_entry.insert(0, "1.0")
            self.param2_label.grid()
            self.param2_entry.grid()
        elif selection == "Realce":
            self.param1_label.grid_remove()
            self.param1_entry.grid_remove()
            self.param2_label.grid_remove()
            self.param2_entry.grid_remove()
    
    def apply_filter(self):
        if self.app.current_image is None:
            self._show_warning()
            return
        
        filter_type = self.filter_var.get()
        
        if filter_type == "Media":
            try:
                k = int(self.param1_entry.get())
                k = max(3, k + 1 if k % 2 == 0 else k)
            except:
                k = 3
            self.app.processed_image = mean_filter(self.app.current_image, k)
        
        elif filter_type == "Mediana":
            try:
                k = int(self.param1_entry.get())
                k = max(3, k + 1 if k % 2 == 0 else k)
            except:
                k = 3
            self.app.processed_image = median_filter(self.app.current_image, k)
        
        elif filter_type == "Mediana Ponderada":
            try:
                n = int(self.param1_entry.get())
                n = max(2, n if n % 2 == 1 else n + 1)
            except:
                n = 3
            
            weights_str = self.param2_entry.get()
            weights = self._parse_weights(weights_str, n)
            
            if weights is None:
                weights = np.array([[1,1,1],[1,3,1],[1,1,1]], dtype=np.int32)
            
            self.app.processed_image = weighted_median_filter(self.app.current_image, weights)
        
        elif filter_type == "Gauss":
            try:
                sigma = float(self.param2_entry.get())
            except:
                sigma = 1.0
            self.app.processed_image = gaussian_filter(self.app.current_image, sigma)
        
        elif filter_type == "Realce":
            self.app.processed_image = edge_enhancement_filter(self.app.current_image)
        
        self.update_display()
    
    def _show_warning(self):
        import tkinter.messagebox as msgbox
        msgbox.showwarning("Advertencia", "Cargar una imagen primero desde Inicio.")
    
    def update_display(self):
        orig_img = self.app.current_image
        proc_img = self.app.get_current_image()
        
        if orig_img is not None:
            h, w = orig_img.shape
            orig_ctk = convert_cv_to_ctk(orig_img, size=(min(w, 300), min(h, 250)))
            self.orig_label.configure(image=orig_ctk, text="")
            self.orig_label.image = orig_ctk
            
            if self.hist_orig_canvas:
                self.hist_orig_canvas.get_tk_widget().destroy()
            canvas_orig = plot_histogram(orig_img, self.hist_orig_frame)
            canvas_orig.get_tk_widget().pack(fill="both", expand=True)
            self.hist_orig_canvas = canvas_orig
        
        if proc_img is not None:
            h, w = proc_img.shape
            proc_ctk = convert_cv_to_ctk(proc_img, size=(min(w, 300), min(h, 250)))
            self.proc_label.configure(image=proc_ctk, text="")
            self.proc_label.image = proc_ctk
            
            if self.hist_proc_canvas:
                self.hist_proc_canvas.get_tk_widget().destroy()
            canvas_proc = plot_histogram(proc_img, self.hist_proc_frame)
            canvas_proc.get_tk_widget().pack(fill="both", expand=True)
            self.hist_proc_canvas = canvas_proc
    
    def _parse_weights(self, weights_str: str, n: int) -> np.ndarray:
        """Parse weights string into numpy array.
        
        Format: "a,b,c; d,e,f; g,h,i" for 3x3
        """
        try:
            rows = weights_str.split(';')
            weights = []
            for row in rows:
                values = [int(x.strip()) for x in row.split(',')]
                weights.append(values)
            weights = np.array(weights, dtype=np.int32)
            
            if weights.shape[0] != n or weights.shape[1] != n:
                return None
            
            return weights
        except:
            return None