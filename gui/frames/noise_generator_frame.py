"""Noise generator frame."""
import customtkinter as ctk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from processing.noise_generators import (
    generate_gaussian, generate_exponential, generate_rayleigh,
    gaussian_pdf, exponential_pdf, rayleigh_pdf
)


class NoiseGeneratorFrame(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        
        self.title_label = ctk.CTkLabel(self, text="Generadores de Ruido", font=("Arial", 16, "bold"))
        self.title_label.pack(pady=5)
        
        self.controls_frame = ctk.CTkFrame(self)
        self.controls_frame.pack(pady=5)
        
        self.dist_label = ctk.CTkLabel(self.controls_frame, text="Distribución:")
        self.dist_label.grid(row=0, column=0, padx=5, pady=5)
        
        self.dist_var = ctk.StringVar(value="Gaussiana")
        self.dist_menu = ctk.CTkOptionMenu(
            self.controls_frame, variable=self.dist_var,
            values=["Gaussiana", "Exponencial", "Rayleigh"],
            command=self.on_dist_change
        )
        self.dist_menu.grid(row=0, column=1, padx=5, pady=5)
        
        self.param1_label = ctk.CTkLabel(self.controls_frame, text="Media (μ):")
        self.param1_label.grid(row=1, column=0, padx=5, pady=5)
        self.param1_entry = ctk.CTkEntry(self.controls_frame, width=150)
        self.param1_entry.insert(0, "0")
        self.param1_entry.grid(row=1, column=1, padx=5, pady=5)
        
        self.param2_label = ctk.CTkLabel(self.controls_frame, text="Sigma (σ):")
        self.param2_label.grid(row=2, column=0, padx=5, pady=5)
        self.param2_entry = ctk.CTkEntry(self.controls_frame, width=150)
        self.param2_entry.insert(0, "1")
        self.param2_entry.grid(row=2, column=1, padx=5, pady=5)
        
        self.param2_label.grid()
        self.param2_entry.grid()
        
        self.samples_label = ctk.CTkLabel(self.controls_frame, text="Muestras:")
        self.samples_label.grid(row=3, column=0, padx=5, pady=5)
        self.samples_entry = ctk.CTkEntry(self.controls_frame, width=150)
        self.samples_entry.insert(0, "100000")
        self.samples_entry.grid(row=3, column=1, padx=5, pady=5)
        
        self.btn_generate = ctk.CTkButton(
            self.controls_frame, text="Generar y Graficar",
            command=self.generate_and_plot, width=150
        )
        self.btn_generate.grid(row=4, column=0, columnspan=2, pady=10)
        
        self.plot_frame = ctk.CTkFrame(self)
        self.plot_frame.pack(fill="both", expand=True, pady=5)
        
        self.canvas = None
    
    def on_dist_change(self, selection):
        if selection == "Gaussiana":
            self.param1_label.configure(text="Media (μ):")
            self.param1_entry.delete(0, "end")
            self.param1_entry.insert(0, "0")
            self.param2_label.configure(text="Sigma (σ):")
            self.param2_entry.delete(0, "end")
            self.param2_entry.insert(0, "1")
            self.param2_label.grid()
            self.param2_entry.grid()
        elif selection == "Exponencial":
            self.param1_label.configure(text="Lambda (λ):")
            self.param1_entry.delete(0, "end")
            self.param1_entry.insert(0, "1")
            self.param2_label.grid_remove()
            self.param2_entry.grid_remove()
        elif selection == "Rayleigh":
            self.param1_label.configure(text="Xi (ξ):")
            self.param1_entry.delete(0, "end")
            self.param1_entry.insert(0, "1")
            self.param2_label.grid_remove()
            self.param2_entry.grid_remove()
    
    def generate_and_plot(self):
        try:
            n_samples = int(self.samples_entry.get())
            n_samples = max(100, min(n_samples, 1000000))
        except ValueError:
            n_samples = 100000
        
        dist = self.dist_var.get()
        
        if dist == "Gaussiana":
            try:
                mean = float(self.param1_entry.get())
                sigma = float(self.param2_entry.get())
            except ValueError:
                mean, sigma = 0.0, 1.0
            samples = generate_gaussian(n_samples, mean, sigma)
            pdf_func = lambda x: gaussian_pdf(x, mean, sigma)
            x_range = (mean - 4*sigma, mean + 4*sigma)
        elif dist == "Exponencial":
            try:
                lam = float(self.param1_entry.get())
            except ValueError:
                lam = 1.0
            samples = generate_exponential(n_samples, lam)
            pdf_func = lambda x: exponential_pdf(x, lam)
            x_range = (0, np.percentile(samples, 99.9))
        elif dist == "Rayleigh":
            try:
                xi = float(self.param1_entry.get())
            except ValueError:
                xi = 1.0
            samples = generate_rayleigh(n_samples, xi)
            pdf_func = lambda x: rayleigh_pdf(x, xi)
            x_range = (0, np.percentile(samples, 99.9))
        
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        
        fig = Figure(figsize=(6, 4), dpi=100)
        fig.subplots_adjust(left=0.12, bottom=0.12, right=0.95, top=0.90)
        ax = fig.add_subplot(111)
        
        counts, bins = np.histogram(samples, bins=100, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax.bar(bin_centers, counts, width=bins[1]-bins[0], alpha=0.7, color='steelblue', label='Histograma')
        
        x_theory = np.linspace(x_range[0], x_range[1], 500)
        y_theory = pdf_func(x_theory)
        ax.plot(x_theory, y_theory, 'r-', linewidth=2, label='PDF teórica')
        
        ax.set_xlim(x_range[0], x_range[1])
        ax.set_xlabel('Valor', fontsize=10)
        ax.set_ylabel('Densidad', fontsize=10)
        ax.set_title(f'Distribución {dist}\n(n={n_samples:,})', fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.draw()