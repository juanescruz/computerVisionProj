"""Histogram display frame for standalone histogram viewing."""
import customtkinter as ctk

from processing.point_operators import calc_histogram
from gui.utils import plot_histogram


class HistogramFrame(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        
        self.title_label = ctk.CTkLabel(self, text="Visualización de Histograma", font=("Arial", 16, "bold"))
        self.title_label.pack(pady=5)
        
        self.btn_show = ctk.CTkButton(self, text="Mostrar Histograma", command=self.show_histogram, width=150)
        self.btn_show.pack(pady=5)
        
        self.hist_canvas = None
        
        self.hist_frame = ctk.CTkFrame(self)
        self.hist_frame.pack(fill="both", expand=True, pady=5)
    
    def show_histogram(self):
        if self.app.current_image is None:
            return
        self.update_display()
    
    def update_display(self):
        img = self.app.get_current_image()
        
        if self.hist_canvas:
            self.hist_canvas.get_tk_widget().destroy()
        
        canvas = plot_histogram(img, self.hist_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.hist_canvas = canvas