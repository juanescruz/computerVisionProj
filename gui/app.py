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
from gui.frames.edge_detection_frame import EdgeDetectionFrame
from gui.frames.edge_noise_frame import EdgeNoiseFrame
from gui.frames.laplacian_frame import LaplacianFrame
from gui.frames.diffusion_frame import DiffusionFrame
from gui.frames.bilateral_frame import BilateralFrame
from gui.frames.thresholding_auto_frame import ThresholdingAutoFrame
from gui.frames.canny_frame import CannyFrame
from gui.frames.susan_frame import SusanFrame
from gui.frames.hough_frame import HoughFrame
from gui.frames.active_contours_frame import ActiveContoursFrame
from gui.frames.video_tracking_frame import VideoTrackingFrame
from gui.frames.sift_frame import SiftFrame


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("TP1 - Procesamiento de Imágenes")
        self.geometry("1200x800")
        
        self.current_image = None
        self.current_image_color = None
        self.processed_image = None
        
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.tab_home = self.tabview.add("Inicio")
        #self.tab_gamma = self.tabview.add("Gamma")
        #self.tab_negative = self.tabview.add("Negativo")
        #self.tab_threshold = self.tabview.add("Umbral")
        #self.tab_equalize = self.tabview.add("Ecualizar")
        #self.tab_histogram = self.tabview.add("Histograma")
        #self.tab_noise = self.tabview.add("Ruido")
        #self.tab_contamination = self.tabview.add("Contaminar")
        #self.tab_filters = self.tabview.add("Filtros")
        #self.tab_experimentation = self.tabview.add("Experimentación")
        #self.tab_edge_detection = self.tabview.add("Detección de Bordes Clásica")
        #self.tab_edge_noise = self.tabview.add("Ruido + Bordes")
        #self.tab_laplacian = self.tabview.add("Laplaciano con Ruido")
        #self.tab_diffusion = self.tabview.add("Difusión")
        #self.tab_bilateral = self.tabview.add("Filtro Bilateral")
        #self.tab_thresholding_auto = self.tabview.add("Umbralización Automática")
        #self.tab_canny = self.tabview.add("Canny")
        #self.tab_susan = self.tabview.add("SUSAN")
        #self.tab_hough = self.tabview.add("Hough")
        #self.tab_active_contours = self.tabview.add("Contornos Activos")
        #self.tab_video = self.tabview.add("Seguimiento Video")
        self.tab_sift = self.tabview.add("SIFT")
        
        self.frames = {}
        self.frames["home"] = HomeFrame(self.tab_home, self)
        ##self.frames["gamma"] = GammaFrame(self.tab_gamma, self)
        ##self.frames["negative"] = NegativeFrame(self.tab_negative, self)
        ##self.frames["threshold"] = ThresholdFrame(self.tab_threshold, self)
        ##self.frames["equalize"] = EqualizeFrame(self.tab_equalize, self)
        ##elf.frames["histogram"] = HistogramFrame(self.tab_histogram, self)
        ##self.frames["noise"] = NoiseGeneratorFrame(self.tab_noise, self)
        ##self.frames["contamination"] = ContaminationFrame(self.tab_contamination, self)
        ##self.frames["filters"] = SpatialFiltersFrame(self.tab_filters, self)
        ##self.frames["experimentation"] = ExperimentationFrame(self.tab_experimentation, self)
        ##self.frames["edge_detection"] = EdgeDetectionFrame(self.tab_edge_detection, self)
        ##self.frames["edge_noise"] = EdgeNoiseFrame(self.tab_edge_noise, self)
        ##self.frames["laplacian"] = LaplacianFrame(self.tab_laplacian, self)
        ##self.frames["diffusion"] = DiffusionFrame(self.tab_diffusion, self)
        #self.frames["bilateral"] = BilateralFrame(self.tab_bilateral, self)
        #self.frames["thresholding_auto"] = ThresholdingAutoFrame(self.tab_thresholding_auto, self)
        #self.frames["canny"] = CannyFrame(self.tab_canny, self)
        #self.frames["susan"] = SusanFrame(self.tab_susan, self)
        #self.frames["hough"] = HoughFrame(self.tab_hough, self)
        #self.frames["active_contours"] = ActiveContoursFrame(self.tab_active_contours, self)
        #self.frames["video"] = VideoTrackingFrame(self.tab_video, self)
        self.frames["sift"] = SiftFrame(self.tab_sift, self)
        
        for frame in self.frames.values():
            frame.pack(fill="both", expand=True)
        
        self.tabview.set("Inicio")
    
    def load_image(self, filepath: str):
        """Load image from file path."""
        import cv2
        img_color = cv2.imread(filepath, cv2.IMREAD_COLOR)
        if img_color is None:
            raise ValueError(f"Cannot load image: {filepath}")
        img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)
        self.current_image = img_gray
        self.current_image_color = img_color
        self.processed_image = None
        
        for frame in self.frames.values():
            if hasattr(frame, 'update_display'):
                frame.update_display()
    
    def get_current_image(self, mode="gray") -> np.ndarray:
        """Return current image.
        
        mode: 'gray' (default) or 'color'.
        If processed_image exists, return it (always grayscale).
        """
        if self.processed_image is not None:
            return self.processed_image
        if mode == "color" and self.current_image_color is not None:
            return self.current_image_color
        return self.current_image