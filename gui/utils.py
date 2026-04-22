"""Utility functions for GUI."""
import numpy as np
import customtkinter as ctk
from PIL import Image
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def convert_cv_to_ctk(img: np.ndarray, size=None) -> ctk.CTkImage:
    """Convert OpenCV image (grayscale or BGR) to CTkImage."""
    if len(img.shape) == 2:
        h, w = img.shape
    else:
        h, w, c = img.shape
    
    if len(img.shape) == 2:
        img_rgb = np.stack([img, img, img], axis=-1)
    else:
        img_rgb = img[:, :, ::-1]
    
    img_pil = Image.fromarray(img_rgb.astype('uint8'), mode='RGB')
    
    if size is None:
        size = (w, h)
    
    return ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=size)


def plot_histogram(img: np.ndarray, master, width=300, height=250) -> FigureCanvasTkAgg:
    """Embed matplotlib histogram in tkinter frame."""
    from processing.point_operators import calc_histogram
    
    hist = calc_histogram(img)
    
    has_only_extremes = np.sum(hist > 0) <= 2
    y_max = max(hist) * 1.2 if max(hist) > 0 else 0.1
    if has_only_extremes and max(hist) > 0:
        y_max = max(hist) * 1.3
    
    fig = Figure(figsize=(width/100, height/100), dpi=100)
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
    
    ax = fig.add_subplot(111)
    ax.bar(range(256), hist, width=1.0, color='gray')
    ax.set_xlim(-0.5, 255.5)
    ax.set_ylim(0, y_max)
    ax.set_xlabel('Nivel de gris', fontsize=9)
    ax.set_ylabel('Frecuencia relativa', fontsize=9)
    ax.tick_params(axis='both', labelsize=8)
    ax.grid(True, alpha=0.3)
    
    canvas = FigureCanvasTkAgg(fig, master=master)
    canvas.draw()
    canvas.draw_idle()
    return canvas