"""Contamination frame."""
import customtkinter as ctk
import numpy as np

from processing.noise_contamination import (
    add_gaussian_noise, add_exponential_noise, add_rayleigh_noise,
    add_salt_pepper_noise
)
from gui.utils import convert_cv_to_ctk, plot_histogram


class ContaminationFrame(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        
        self.title_label = ctk.CTkLabel(self, text="Contaminar Imagen", font=("Arial", 16, "bold"))
        self.title_label.pack(pady=5)
        
        self.controls_frame = ctk.CTkFrame(self)
        self.controls_frame.pack(pady=5)
        
        self.noise_label = ctk.CTkLabel(self.controls_frame, text="Tipo de ruido:")
        self.noise_label.grid(row=0, column=0, padx=5, pady=5)
        
        self.noise_var = ctk.StringVar(value="Gaussiano")
        self.noise_menu = ctk.CTkOptionMenu(
            self.controls_frame, variable=self.noise_var,
            values=["Gaussiano", "Exponencial", "Rayleigh", "Sal y Pimienta"],
            command=self.on_noise_change
        )
        self.noise_menu.grid(row=0, column=1, padx=5, pady=5)
        
        self.param1_label = ctk.CTkLabel(self.controls_frame, text="Media (μ):")
        self.param1_label.grid(row=1, column=0, padx=5, pady=5)
        self.param1_entry = ctk.CTkEntry(self.controls_frame, width=150)
        self.param1_entry.insert(0, "0")
        self.param1_entry.grid(row=1, column=1, padx=5, pady=5)
        
        self.param2_label = ctk.CTkLabel(self.controls_frame, text="Sigma (σ):")
        self.param2_label.grid(row=2, column=0, padx=5, pady=5)
        self.param2_entry = ctk.CTkEntry(self.controls_frame, width=150)
        self.param2_entry.insert(0, "25")
        self.param2_entry.grid(row=2, column=1, padx=5, pady=5)
        
        self.param2_label.grid()
        self.param2_entry.grid()
        
        self.percent_label = ctk.CTkLabel(self.controls_frame, text="Porcentaje (%):")
        self.percent_label.grid(row=3, column=0, padx=5, pady=5)
        self.percent_slider = ctk.CTkSlider(
            self.controls_frame, from_=0, to=100, number_of_steps=100,
            command=self.on_percent_change
        )
        self.percent_slider.set(10)
        self.percent_slider.grid(row=3, column=1, padx=5, pady=5)
        
        self.percent_value_label = ctk.CTkLabel(self.controls_frame, text="10%")
        self.percent_value_label.grid(row=3, column=2, padx=5, pady=5)
        
        self.btn_apply = ctk.CTkButton(
            self.controls_frame, text="Aplicar Ruido",
            command=self.apply_noise, width=150
        )
        self.btn_apply.grid(row=4, column=0, columnspan=2, pady=10)
        
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
        ctk.CTkLabel(self.right_frame, text="Contaminada", font=("Arial", 12)).pack(pady=2)
        
        self.hist_orig_frame = ctk.CTkFrame(self.left_frame)
        self.hist_orig_frame.pack(fill="both", expand=True, pady=5)
        
        self.hist_proc_frame = ctk.CTkFrame(self.right_frame)
        self.hist_proc_frame.pack(fill="both", expand=True, pady=5)
        
        self.hist_orig_canvas = None
        self.hist_proc_canvas = None
    
    def on_noise_change(self, selection):
        if selection == "Gaussiano":
            self.param1_label.configure(text="Media (μ):")
            self.param1_entry.delete(0, "end")
            self.param1_entry.insert(0, "0")
            self.param2_label.configure(text="Sigma (σ):")
            self.param2_entry.delete(0, "end")
            self.param2_entry.insert(0, "25")
            self.param1_label.grid()
            self.param1_entry.grid()
            self.param2_label.grid()
            self.param2_entry.grid()
            self.percent_label.grid()
            self.percent_slider.grid()
            self.percent_value_label.grid()
        elif selection == "Exponencial":
            self.param1_label.configure(text="Lambda (λ):")
            self.param1_entry.delete(0, "end")
            self.param1_entry.insert(0, "0.05")
            self.param1_label.grid()
            self.param1_entry.grid()
            self.param2_label.grid_remove()
            self.param2_entry.grid_remove()
            self.percent_label.grid()
            self.percent_slider.grid()
            self.percent_value_label.grid()
        elif selection == "Rayleigh":
            self.param1_label.configure(text="Xi (ξ):")
            self.param1_entry.delete(0, "end")
            self.param1_entry.insert(0, "1.2")
            self.param1_label.grid()
            self.param1_entry.grid()
            self.param2_label.grid_remove()
            self.param2_entry.grid_remove()
            self.percent_label.grid()
            self.percent_slider.grid()
            self.percent_value_label.grid()
        elif selection == "Sal y Pimienta":
            self.param1_label.configure(text="Prob. p:")
            self.param1_entry.delete(0, "end")
            self.param1_entry.insert(0, "0.05")
            self.param1_label.grid()
            self.param1_entry.grid()
            self.param2_label.grid_remove()
            self.param2_entry.grid_remove()
            self.percent_label.grid_remove()
            self.percent_slider.grid_remove()
            self.percent_value_label.grid_remove()
    
    def on_percent_change(self, value):
        self.percent_value_label.configure(text=f"{int(value)}%")
    
    def apply_noise(self):
        if self.app.current_image is None:
            self._show_warning()
            return
        
        try:
            percentage = self.percent_slider.get() / 100.0
        except:
            percentage = 0.1
        
        noise_type = self.noise_var.get()
        
        if noise_type == "Gaussiano":
            try:
                mean = float(self.param1_entry.get())
                sigma = float(self.param2_entry.get())
            except ValueError:
                mean, sigma = 0.0, 25.0
            self.app.processed_image = add_gaussian_noise(
                self.app.current_image, percentage, mean, sigma
            )
        elif noise_type == "Exponencial":
            try:
                lam = float(self.param1_entry.get())
            except ValueError:
                lam = 0.05
            self.app.processed_image = add_exponential_noise(
                self.app.current_image, percentage, lam
            )
        elif noise_type == "Rayleigh":
            try:
                xi = float(self.param1_entry.get())
            except ValueError:
                xi = 1.2
            self.app.processed_image = add_rayleigh_noise(
                self.app.current_image, percentage, xi
            )
        elif noise_type == "Sal y Pimienta":
            try:
                p = float(self.param1_entry.get())
            except ValueError:
                p = 0.05
            self.app.processed_image = add_salt_pepper_noise(self.app.current_image, p)
        
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