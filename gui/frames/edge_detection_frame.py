"""Edge detection frame."""
import customtkinter as ctk

from processing.edge_detection import (
    prewitt_operator,
    sobel_operator
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

        ctk.CTkLabel(
            self.right_frame,
            text="Bordes",
            font=("Arial", 12)
        ).pack(pady=2)

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

    def apply_prewitt(self):

        if self.app.current_image is None:
            return

        img = self.app.get_current_image()

        self.app.processed_image = prewitt_operator(img)

        self.update_display()

    def apply_sobel(self):

        if self.app.current_image is None:
            return

        img = self.app.get_current_image()

        self.app.processed_image = sobel_operator(img)

        self.update_display()
        
    def reset_image(self):

        self.app.processed_image = None
        self.update_display()

    def update_display(self):

        orig_img = self.app.current_image

        proc_img = self.app.get_current_image()

        if orig_img is not None:

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

            canvas_orig = plot_histogram(
                orig_img,
                self.hist_orig_frame
            )

            canvas_orig.get_tk_widget().pack(
                fill="both",
                expand=True
            )

            self.hist_orig_canvas = canvas_orig

        if proc_img is not None:

            h, w = proc_img.shape

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

            canvas_proc = plot_histogram(
                proc_img,
                self.hist_proc_frame
            )

            canvas_proc.get_tk_widget().pack(
                fill="both",
                expand=True
            )

            self.hist_proc_canvas = canvas_proc