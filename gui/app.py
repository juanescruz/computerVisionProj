"""Main application with CTkTabview."""
import customtkinter as ctk
import numpy as np
from tkinter import filedialog

from gui.frames.home_frame import HomeFrame
from gui.frames.gamma_frame import GammaFrame
from gui.frames.negative_frame import NegativeFrame
from gui.frames.threshold_frame import ThresholdFrame
from gui.frames.equalize_frame import EqualizeFrame
from gui.frames.histogram_frame import HistogramFrame
from gui.frames.noise_generator_frame import NoiseGeneratorFrame
from gui.frames.contamination_frame import ContaminationFrame
from gui.frames.spatial_filters_frame import SpatialFiltersFrame
from gui.frames.experimentation_frame import ExperimentationFrame


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("TP1 - Procesamiento de Imágenes")
        self.geometry("1200x800")
        
        self.current_image = None
        self.processed_image = None
        
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.tab_home = self.tabview.add("Inicio")
        self.tab_gamma = self.tabview.add("Gamma")
        self.tab_negative = self.tabview.add("Negativo")
        self.tab_threshold = self.tabview.add("Umbral")
        self.tab_equalize = self.tabview.add("Ecualizar")
        self.tab_histogram = self.tabview.add("Histograma")
        self.tab_noise = self.tabview.add("Ruido")
        self.tab_contamination = self.tabview.add("Contaminar")
        self.tab_filters = self.tabview.add("Filtros")
        self.tab_experimentation = self.tabview.add("Experimentación")
        
        self.frames = {}
        self.frames["home"] = HomeFrame(self.tab_home, self)
        self.frames["gamma"] = GammaFrame(self.tab_gamma, self)
        self.frames["negative"] = NegativeFrame(self.tab_negative, self)
        self.frames["threshold"] = ThresholdFrame(self.tab_threshold, self)
        self.frames["equalize"] = EqualizeFrame(self.tab_equalize, self)
        self.frames["histogram"] = HistogramFrame(self.tab_histogram, self)
        self.frames["noise"] = NoiseGeneratorFrame(self.tab_noise, self)
        self.frames["contamination"] = ContaminationFrame(self.tab_contamination, self)
        self.frames["filters"] = SpatialFiltersFrame(self.tab_filters, self)
        self.frames["experimentation"] = ExperimentationFrame(self.tab_experimentation, self)
        
        for frame in self.frames.values():
            frame.pack(fill="both", expand=True)
        
        self.tabview.set("Inicio")
    
    def load_image(self, filepath: str):
        """Load image from file path."""
        import cv2
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot load image: {filepath}")
        self.current_image = img
        self.processed_image = None
        
        for frame in self.frames.values():
            if hasattr(frame, 'update_display'):
                frame.update_display()
    
    def get_current_image(self) -> np.ndarray:
        """Return current image (original or processed)."""
        return self.processed_image if self.processed_image is not None else self.current_image