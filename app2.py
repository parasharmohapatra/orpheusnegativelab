import os
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from skimage.io import imread, imsave
from skimage.exposure import match_histograms, adjust_gamma, rescale_intensity
from skimage.color import rgb2lab, lab2rgb
import numpy as np
from PIL import Image, ImageTk

class ImageEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Editor")
        
        # Configure style
        style = ttk.Style()
        style.configure('TFrame', padding=5)
        style.configure('TButton', padding=2)
        style.configure('TLabel', padding=2)

        # Create main container
        self.main_container = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel for controls
        self.controls_frame = ttk.Frame(self.main_container)
        self.main_container.add(self.controls_frame, weight=1)

        # Right panel for image
        self.image_frame = ttk.Frame(self.main_container)
        self.main_container.add(self.image_frame, weight=3)

        # Initialize variables
        self.image_files = []
        self.current_index = 0
        self.current_image = None
        self.edited_image = None
        self.base_adjusted_image = None
        self.rotation_count = 0  # Track the number of 90-degree rotations
        self.is_flipped = False  # Track whether the image is flipped

        self._create_controls()
        self._create_canvas()

    def _create_controls(self):
        # File controls section
        file_section = ttk.LabelFrame(self.controls_frame, text="File Controls", padding=5)
        file_section.pack(fill=tk.X, padx=5, pady=5)

        self.open_button = ttk.Button(file_section, text="Open Directory", command=self.open_directory)
        self.open_button.pack(fill=tk.X, pady=2)

        nav_frame = ttk.Frame(file_section)
        nav_frame.pack(fill=tk.X, pady=2)
        self.prev_button = ttk.Button(nav_frame, text="Previous", command=self.previous_image)
        self.prev_button.pack(side=tk.LEFT, expand=True, padx=2)
        self.next_button = ttk.Button(nav_frame, text="Next", command=self.next_image)
        self.next_button.pack(side=tk.LEFT, expand=True, padx=2)

        self.save_button = ttk.Button(file_section, text="Save", command=self.save_image)
        self.save_button.pack(fill=tk.X, pady=2)

        # Image controls section
        image_section = ttk.LabelFrame(self.controls_frame, text="Image Controls", padding=5)
        image_section.pack(fill=tk.X, padx=5, pady=5)

        transform_frame = ttk.Frame(image_section)
        transform_frame.pack(fill=tk.X, pady=2)
        self.rotate_button = ttk.Button(transform_frame, text="Rotate", command=self.rotate_image)
        self.rotate_button.pack(side=tk.LEFT, expand=True, padx=2)
        self.flip_button = ttk.Button(transform_frame, text="Flip", command=self.flip_image)
        self.flip_button.pack(side=tk.LEFT, expand=True, padx=2)

        # Reset button
        self.reset_button = ttk.Button(image_section, text="Reset", command=self.reset_edits)
        self.reset_button.pack(fill=tk.X, pady=2)

        # Adjustments section
        adjust_section = ttk.LabelFrame(self.controls_frame, text="Adjustments", padding=5)
        adjust_section.pack(fill=tk.X, padx=5, pady=5)

        # Gamma control
        ttk.Label(adjust_section, text="Gamma").pack(anchor=tk.W)
        self.gamma_slider = ttk.Scale(adjust_section, from_=0.1, to=10.0, orient=tk.HORIZONTAL)
        self.gamma_slider.set(4.0)
        self.gamma_slider.pack(fill=tk.X, pady=(0, 5))

        # Exposure control
        ttk.Label(adjust_section, text="Exposure").pack(anchor=tk.W)
        self.exposure_slider = ttk.Scale(adjust_section, from_=0.5, to=3.0, orient=tk.HORIZONTAL)
        self.exposure_slider.set(1.0)
        self.exposure_slider.pack(fill=tk.X, pady=(0, 5))

        # Color balance section
        color_section = ttk.LabelFrame(self.controls_frame, text="Color Balance", padding=5)
        color_section.pack(fill=tk.X, padx=5, pady=5)

        # Red balance
        ttk.Label(color_section, text="Red").pack(anchor=tk.W)
        self.r_slider = ttk.Scale(color_section, from_=0.5, to=2.0, orient=tk.HORIZONTAL)
        self.r_slider.set(1.0)
        self.r_slider.pack(fill=tk.X, pady=(0, 5))

        # Green balance
        ttk.Label(color_section, text="Green").pack(anchor=tk.W)
        self.g_slider = ttk.Scale(color_section, from_=0.5, to=2.0, orient=tk.HORIZONTAL)
        self.g_slider.set(1.0)
        self.g_slider.pack(fill=tk.X, pady=(0, 5))

        # Blue balance
        ttk.Label(color_section, text="Blue").pack(anchor=tk.W)
        self.b_slider = ttk.Scale(color_section, from_=0.5, to=2.0, orient=tk.HORIZONTAL)
        self.b_slider.set(1.0)
        self.b_slider.pack(fill=tk.X, pady=(0, 5))

        # Add slider update bindings
        for slider in [self.gamma_slider, self.exposure_slider, 
                      self.r_slider, self.g_slider, self.b_slider]:
            slider.bind("<ButtonRelease-1>", self.update_image)

    def _create_canvas(self):
        # Create canvas with scrollbars
        self.canvas = tk.Canvas(self.image_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def open_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.image_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith((".cr2", ".cr3", ".dng"))]
            self.current_index = 0
            if self.image_files:
                self.load_image()

    def load_image(self):
        file_path = self.image_files[self.current_index]
        self.current_image = self.read_raw_image(file_path)
        self.reset_edits()  # Reset sliders and apply initial processing
        self.display_image()

    def read_raw_image(self, file_path):
        import rawpy
        with rawpy.imread(file_path) as raw:
            return raw.postprocess()

    def invert_colors(self, image):
        return 255 - image

    def apply_initial_processing(self):
        # Step 1: Apply histogram matching
        self.base_adjusted_image = self.current_image.copy()
        self.base_adjusted_image[:, :, 0] = match_histograms(self.current_image[:, :, 0], self.current_image[:, :, 1])
        self.base_adjusted_image[:, :, 2] = match_histograms(self.current_image[:, :, 2], self.current_image[:, :, 1])
        
        # Step 2: Invert colors
        self.base_adjusted_image = self.invert_colors(self.base_adjusted_image)
        
        # Step 3: Apply gamma correction to inverted image
        gamma = self.gamma_slider.get()
        self.base_adjusted_image = adjust_gamma(self.base_adjusted_image, gamma=gamma)

        # Apply rotation and flip transformations
        if self.rotation_count > 0:
            self.base_adjusted_image = np.rot90(self.base_adjusted_image, k=self.rotation_count)
        if self.is_flipped:
            self.base_adjusted_image = np.fliplr(self.base_adjusted_image)

    def apply_adjustments(self):
        self.edited_image = self.base_adjusted_image.copy()
        
        # Apply color balance
        r_balance = self.r_slider.get()
        g_balance = self.g_slider.get()
        b_balance = self.b_slider.get()
        
        self.edited_image[:,:,0] = np.clip(self.edited_image[:,:,0] * r_balance, 0, 255)
        self.edited_image[:,:,1] = np.clip(self.edited_image[:,:,1] * g_balance, 0, 255)
        self.edited_image[:,:,2] = np.clip(self.edited_image[:,:,2] * b_balance, 0, 255)

        # Apply exposure adjustment
        exposure = self.exposure_slider.get()
        self.edited_image = np.clip(self.edited_image * exposure, 0, 255)

    def update_image(self, event=None):
        if self.current_image is not None:
            if event and event.widget == self.gamma_slider:
                self.apply_initial_processing()
            self.apply_adjustments()
            self.display_image()

    def display_image(self):
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1:  # Canvas not yet realized
            canvas_width = 800
            canvas_height = 600

        img = Image.fromarray(self.edited_image.astype(np.uint8))
        
        # Calculate scaling to fit canvas while maintaining aspect ratio
        img_ratio = img.size[0] / img.size[1]
        canvas_ratio = canvas_width / canvas_height
        
        if img_ratio > canvas_ratio:
            new_width = canvas_width
            new_height = int(canvas_width / img_ratio)
        else:
            new_height = canvas_height
            new_width = int(canvas_height * img_ratio)

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(img)
        
        # Clear previous image and create new one centered
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width//2, canvas_height//2,
            image=self.tk_image,
            anchor=tk.CENTER
        )

    def save_image(self):
        file_path = self.image_files[self.current_index]
        save_path = os.path.splitext(file_path)[0] + "_edited.jpg"
        imsave(save_path, self.edited_image.astype(np.uint8))

    def previous_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image()

    def next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_image()

    def rotate_image(self):
        self.rotation_count = (self.rotation_count + 1) % 4
        self.apply_initial_processing()
        self.apply_adjustments()
        self.display_image()

    def flip_image(self):
        self.is_flipped = not self.is_flipped
        self.apply_initial_processing()
        self.apply_adjustments()
        self.display_image()

    def reset_edits(self):
        # Reset sliders to default values
        self.gamma_slider.set(1.0)
        self.exposure_slider.set(1.0)
        self.r_slider.set(1.0)
        self.g_slider.set(1.0)
        self.b_slider.set(1.0)

        # Reset rotation and flip state
        self.rotation_count = 0
        self.is_flipped = False

        # Reapply initial processing and adjustments
        self.apply_initial_processing()
        self.apply_adjustments()
        self.display_image()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1200x800")  # Set initial window size
    app = ImageEditorApp(root)
    root.mainloop()
