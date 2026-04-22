"""Gamma correction frame."""
import customtkinter as ctk

from processing.point_operators import gamma_correction
from gui.utils import convert_cv_to_ctk, plot_histogram


class GammaFrame(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        
        self.title_label = ctk.CTkLabel(self, text="Corrección Gamma", font=("Arial", 16, "bold"))
        self.title_label.pack(pady=5)
        
        self.controls_frame = ctk.CTkFrame(self)
        self.controls_frame.pack(pady=5)
        
        self.gamma_label = ctk.CTkLabel(self.controls_frame, text="Gamma: 1.0")
        self.gamma_label.pack(pady=5)
        
        self.gamma_slider = ctk.CTkSlider(
            self.controls_frame, from_=0.1, to=2.0, number_of_steps=19,
            command=self.on_slider_change
        )
        self.gamma_slider.set(1.0)
        self.gamma_slider.pack(pady=5)
        
        self.btn_apply = ctk.CTkButton(self.controls_frame, text="Aplicar", command=self.apply_gamma, width=150)
        self.btn_apply.pack(pady=5)
        
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
        ctk.CTkLabel(self.right_frame, text="Procesada", font=("Arial", 12)).pack(pady=2)
        
        self.hist_orig_frame = ctk.CTkFrame(self.left_frame)
        self.hist_orig_frame.pack(fill="both", expand=True, pady=5)
        
        self.hist_proc_frame = ctk.CTkFrame(self.right_frame)
        self.hist_proc_frame.pack(fill="both", expand=True, pady=5)
        
        self.hist_orig_canvas = None
        self.hist_proc_canvas = None
    
    def on_slider_change(self, value):
        self.gamma_label.configure(text=f"Gamma: {value:.2f}")
    
    def apply_gamma(self):
        if self.app.current_image is None:
            return
        
        gamma = self.gamma_slider.get()
        self.app.processed_image = gamma_correction(self.app.current_image, gamma)
        self.update_display()
    
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