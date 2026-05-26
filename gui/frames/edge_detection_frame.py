"""Edge detection frame."""
import customtkinter as ctk
import numpy as np

from processing.edge_detection import (
    prewitt_operator,
    sobel_operator,
    prewitt_color,
    sobel_color
)

from gui.utils import (
    convert_cv_to_ctk,
    plot_histogram
)


class EdgeDetectionFrame(ctk.CTkFrame):

    def __init__(self, parent, app):
        super().__init__(parent)

        self.app = app

        self.title_label = ctk.CTkLabel(
            self,
            text="Detector de Bordes",
            font=("Arial", 16, "bold")
        )
        self.title_label.pack(pady=5)

        self.button_frame = ctk.CTkFrame(self)
        self.button_frame.pack(pady=5)

        self.btn_prewitt = ctk.CTkButton(
            self.button_frame,
            text="Prewitt",
            command=self.apply_prewitt,
            width=120
        )
        self.btn_prewitt.pack(side="left", padx=5)

        self.btn_sobel = ctk.CTkButton(
            self.button_frame,
            text="Sobel",
            command=self.apply_sobel,
            width=120
        )
        self.btn_sobel.pack(side="left", padx=5)
        
        self.btn_reset = ctk.CTkButton(
            self.button_frame,
            text="Reset",
            command=self.reset_image,
            width=120
        )
        self.btn_reset.pack(side="left", padx=5)

        self.mode_label = ctk.CTkLabel(
            self.button_frame, text="Modo:"
        )
        self.mode_label.pack(side="left", padx=(10, 2))

        self.mode_var = ctk.StringVar(value="Grises")
        self.mode_menu = ctk.CTkOptionMenu(
            self.button_frame,
            values=["Grises", "Color"],
            variable=self.mode_var,
            width=140
        )
        self.mode_menu.pack(side="left", padx=5)

        self.comparison_frame = ctk.CTkFrame(self)
        self.comparison_frame.pack(
            fill="both",
            expand=True,
            pady=5
        )

        self.left_frame = ctk.CTkFrame(
            self.comparison_frame
        )
        self.left_frame.pack(
            side="left",
            fill="both",
            expand=True,
            padx=5
        )

        self.right_frame = ctk.CTkFrame(
            self.comparison_frame
        )
        self.right_frame.pack(
            side="right",
            fill="both",
            expand=True,
            padx=5
        )

        self.orig_label = ctk.CTkLabel(
            self.left_frame,
            text="Sin imagen",
            text_color="gray"
        )
        self.orig_label.pack(pady=5)

        ctk.CTkLabel(
            self.left_frame,
            text="Original",
            font=("Arial", 12)
        ).pack(pady=2)

        self.proc_label = ctk.CTkLabel(
            self.right_frame,
            text="Sin resultado",
            text_color="gray"
        )
        self.proc_label.pack(pady=5)

        self.proc_title = ctk.CTkLabel(
            self.right_frame,
            text="Bordes",
            font=("Arial", 12)
        )
        self.proc_title.pack(pady=2)

        self.hist_orig_frame = ctk.CTkFrame(
            self.left_frame
        )
        self.hist_orig_frame.pack(
            fill="both",
            expand=True,
            pady=5
        )

        self.hist_proc_frame = ctk.CTkFrame(
            self.right_frame
        )
        self.hist_proc_frame.pack(
            fill="both",
            expand=True,
            pady=5
        )

        self.hist_orig_canvas = None
        self.hist_proc_canvas = None

    def _select_operator(self, operator_fn, color_fn):
        if self.app.current_image is None:
            return

        mode = self.mode_var.get()

        if mode == "Grises":
            img = self.app.get_current_image(mode="gray")
            self.app.processed_image = operator_fn(img)
        else:
            img = self.app.get_current_image(mode="color")
            if img is None or len(img.shape) != 3:
                import tkinter.messagebox as msgbox
                msgbox.showwarning("Advertencia", "No hay imagen color cargada.")
                return
            method = "euclidean" if mode == "Color" else "max"
            self.app.processed_image = color_fn(img, method=method)

        self.update_display()

    def apply_prewitt(self):
        self._select_operator(prewitt_operator, prewitt_color)

    def apply_sobel(self):
        self._select_operator(sobel_operator, sobel_color)
        
    def reset_image(self):

        self.app.processed_image = None
        self.update_display()

    def plot_rgb_histogram(self, img, parent_frame):
        """Histograma RGB con 3 canales superpuestos."""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        R = img[:, :, 0].ravel()
        G = img[:, :, 1].ravel()
        B = img[:, :, 2].ravel()

        fig = Figure(figsize=(3, 2.5), dpi=80)
        ax = fig.add_subplot(111)
        ax.hist(R, bins=256, range=(0, 255), color='red', alpha=0.5, label='R')
        ax.hist(G, bins=256, range=(0, 255), color='green', alpha=0.5, label='G')
        ax.hist(B, bins=256, range=(0, 255), color='blue', alpha=0.5, label='B')
        ax.set_xlim(0, 255)
        ax.set_xlabel("Intensidad", fontsize=8)
        ax.set_ylabel("Frecuencia", fontsize=8)
        ax.legend(loc='upper right', fontsize=7)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        return canvas

    def update_display(self):

        mode = self.mode_var.get()
        is_color_mode = mode != "Grises" and self.app.current_image_color is not None

        if is_color_mode:
            orig_img = self.app.current_image_color
        else:
            orig_img = self.app.current_image

        proc_img = self.app.processed_image

        if orig_img is not None:

            if len(orig_img.shape) == 3:
                h, w, _ = orig_img.shape
            else:
                h, w = orig_img.shape

            orig_ctk = convert_cv_to_ctk(
                orig_img,
                size=(min(w, 300), min(h, 250))
            )

            self.orig_label.configure(
                image=orig_ctk,
                text=""
            )

            self.orig_label.image = orig_ctk

            if self.hist_orig_canvas:
                self.hist_orig_canvas.get_tk_widget().destroy()

            if is_color_mode:
                canvas_orig = self.plot_rgb_histogram(orig_img, self.hist_orig_frame)
            else:
                canvas_orig = plot_histogram(orig_img, self.hist_orig_frame)

            canvas_orig.get_tk_widget().pack(
                fill="both",
                expand=True
            )

            self.hist_orig_canvas = canvas_orig

        self.proc_title.configure(
            text="Bordes Color" if is_color_mode else "Bordes"
        )

        if proc_img is not None:

            is_proc_color = len(proc_img.shape) == 3
            h, w = proc_img.shape[:2]

            proc_ctk = convert_cv_to_ctk(
                proc_img,
                size=(min(w, 300), min(h, 250))
            )

            self.proc_label.configure(
                image=proc_ctk,
                text=""
            )

            self.proc_label.image = proc_ctk

            if self.hist_proc_canvas:
                self.hist_proc_canvas.get_tk_widget().destroy()

            if is_proc_color:
                canvas_proc = self.plot_rgb_histogram(proc_img, self.hist_proc_frame)
            else:
                canvas_proc = plot_histogram(proc_img, self.hist_proc_frame)

            canvas_proc.get_tk_widget().pack(
                fill="both",
                expand=True
            )

            self.hist_proc_canvas = canvas_proc