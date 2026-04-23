"""Home frame for loading images."""
import customtkinter as ctk
from tkinter import filedialog
import cv2
import numpy as np
import os

from gui.utils import convert_cv_to_ctk, plot_histogram

RAW_SIZES = {
    "GIRL.RAW": (164, 389),
    "BARCO.RAW": (207, 290),
    "LENA.RAW": (256, 256),
    "LENAX.RAW": (256, 256),
    "GIRL2.RAW": (256, 256),
    "FRACTAL.RAW": (200, 200),
}


class HomeFrame(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        
        self.title_label = ctk.CTkLabel(self, text="Cargar Imagen", font=("Arial", 20, "bold"))
        self.title_label.pack(pady=10)
        
        self.btn_load = ctk.CTkButton(self, text="Cargar Imagen", command=self.load_image, width=200)
        self.btn_load.pack(pady=10)
        
        self.image_frame = ctk.CTkFrame(self)
        self.image_frame.pack(fill="both", expand=True, pady=10)
        
        self.image_label = ctk.CTkLabel(self.image_frame, text="No hay imagen cargada", text_color="gray")
        self.image_label.pack(pady=5)
        
        self.hist_frame = ctk.CTkFrame(self)
        self.hist_frame.pack(fill="both", expand=True, pady=5)
        
        self.hist_canvas = None
        
        self.btn_verify = ctk.CTkButton(self, text="Verificar Carga", command=self.verify_loaded, width=200)
        self.btn_verify.pack(pady=5)
        
        self.status_label = ctk.CTkLabel(self, text="", text_color="gray")
        self.status_label.pack(pady=5)
    
    def load_image(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.bmp *.raw *.tif *.tiff"), ("All files", "*.*")]
        )

        if filepath:
            try:
                filename = os.path.basename(filepath).upper()

                if filepath.lower().endswith(".raw"):
                    if filename not in RAW_SIZES:
                        raise ValueError("Unknown dimensions for this .raw file")

                    height, width = RAW_SIZES[filename]

                    img = np.fromfile(filepath, dtype=np.uint8)
                    img = img.reshape((height, width))

                else:
                    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    raise ValueError("Cannot decode image")

                self.app.current_image = img
                self.app.processed_image = None
                self.status_label.configure(text=f"Cargado: {filename}")
                self.update_display()

            except Exception as e:
                self.status_label.configure(text=f"Error: {str(e)}")
    
    def verify_loaded(self):
        if self.app.current_image is not None:
            img = self.app.current_image
            self.status_label.configure(
                text=f"Imagen: shape={img.shape}, dtype={img.dtype}, min={img.min()}, max={img.max()}"
            )
        else:
            self.status_label.configure(text="Sin imagen cargada")
    
    def update_display(self):
        if self.app.current_image is not None:
            img = self.app.current_image
            h, w = img.shape
            ctk_img = convert_cv_to_ctk(img, size=(min(w, 500), min(h, 400)))
            self.image_label.configure(image=ctk_img, text="")
            self.image_label.image = ctk_img
            
            if self.hist_canvas:
                self.hist_canvas.get_tk_widget().destroy()
            
            canvas = plot_histogram(img, self.hist_frame, width=500, height=300)
            canvas.get_tk_widget().pack(fill="both", expand=True)
            self.hist_canvas = canvas
        else:
            self.image_label.configure(text="No hay imagen cargada", text_color="gray")