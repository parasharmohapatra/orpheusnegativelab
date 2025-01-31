import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import rawpy
import numpy as np
import os
import yaml
import cProfile
import pstats

def process_raw_image(file_path):
    with rawpy.imread(file_path) as raw:
        raw_image = raw.raw_image_visible.copy()
        rgb_image = raw.postprocess()

    r = raw_image[0::2, 0::2]
    g1 = raw_image[0::2, 1::2]
    g2 = raw_image[1::2, 0::2]
    b = raw_image[1::2, 1::2]

    g_min_shape = (min(g1.shape[0], g2.shape[0]), min(g1.shape[1], g2.shape[1]))
    g1 = g1[:g_min_shape[0], :g_min_shape[1]]
    g2 = g2[:g_min_shape[0], :g_min_shape[1]]
    g = (g1 + g2) / 2

    return rgb_image, r, g, b

def find_corners(data, bins=16384, left_percentage=0.02, right_percentage=0.02):
    hist, bin_edges = np.histogram(data.ravel(), bins=bins)
    max_value = np.max(hist)
    
    left_threshold = max_value * left_percentage
    right_threshold = max_value * right_percentage
    
    left_corner = np.min(bin_edges[np.where(hist > left_threshold)])
    right_corner = np.max(bin_edges[np.where(hist > right_threshold)])
    
    return left_corner, right_corner

def linear_tone_curve(value, left, right, alpha=0.5):
    if value < left:
        return 255
    elif value > right:
        return 0
    else:
        temp_value = 255 - 255 * (value - left) / (right - left)
        adjusted_value = temp_value - (temp_value * alpha * (255 - temp_value) / 255)
        return max(adjusted_value, 0)

def adjust_contrast(image, contrast_factor):
    def contrast(c):
        factor = (259 * (contrast_factor + 255)) / (255 * (259 - contrast_factor))
        return 128 + factor * (c - 128)

    return image.point(contrast)

def adjust_white_balance(image, temperature, tint=0):
    r, g, b = image.split()
    
    if temperature > 0:
        r_factor = 1 + (temperature / 100)
        b_factor = 1 - (temperature / 200)
        g_factor = 1 + (temperature / 400)
    else:
        r_factor = 1 + (temperature / 200)
        b_factor = 1 - (temperature / 100)
        g_factor = 1 + (temperature / 400)
    
    if tint > 0:
        g_factor *= 1 - (tint / 200)
        r_factor *= 1 + (tint / 400)
        b_factor *= 1 + (tint / 400)
    else:
        g_factor *= 1 - (tint / 200)
        r_factor *= 1 + (tint / 400)
        b_factor *= 1 + (tint / 400)
    
    r = r.point(lambda x: min(255, int(x * r_factor)))
    g = g.point(lambda x: min(255, int(x * g_factor)))
    b = b.point(lambda x: min(255, int(x * b_factor)))
    
    return Image.merge("RGB", (r, g, b))

def apply_tone_curves(rgb_image, r, g, b, alpha_r=0.0, alpha_g=0.0, alpha_b=0.0):
    image_8bit_pil = Image.fromarray(rgb_image)
    r, g, b = image_8bit_pil.split()

    r_left, r_right = find_corners(np.array(r), left_percentage=0.01, right_percentage=0.01)
    g_left, g_right = find_corners(np.array(g), left_percentage=0.01, right_percentage=0.01)
    b_left, b_right = find_corners(np.array(b), left_percentage=0.01, right_percentage=0.01)
    
    tone_curve_data = {
        "r": (r_left, r_right),
        "g": (g_left, g_right),
        "b": (b_left, b_right)
    }
    return tone_curve_data, (r_left, r_right, g_left, g_right, b_left, b_right), Image.merge("RGB", (
        r.point(lambda i: linear_tone_curve(i, r_left, r_right, alpha_r)),
        g.point(lambda i: linear_tone_curve(i, g_left, g_right, alpha_g)),
        b.point(lambda i: linear_tone_curve(i, b_left, b_right, alpha_b)),
    ))

def adjust_exposure(image, exposure_factor):
    multiplier = 2 ** (exposure_factor / 100)
    
    r, g, b = image.split()
    r = r.point(lambda x: min(255, int(x * multiplier)))
    g = g.point(lambda x: min(255, int(x * multiplier)))
    b = b.point(lambda x: min(255, int(x * multiplier)))
    
    return Image.merge("RGB", (r, g, b))

class RawImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Orpheus Negative Lab")
        self.root.geometry("1200x800")
        
        style = ttk.Style()
        if 'vista' in style.theme_names():
            style.theme_use('vista')
        elif 'clam' in style.theme_names():
            style.theme_use('clam')
            
        style.configure('Header.TLabel', font=('Segoe UI', 11, 'bold'))
        style.configure('Control.TFrame', padding=10)
        style.configure('Tool.TButton', padding=5)
        
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=4)
        self.root.grid_rowconfigure(0, weight=1)

        self.controls_frame = ttk.Frame(root, padding="10")
        self.controls_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.image_frame = ttk.Frame(root, padding="10")
        self.image_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.image_frame.grid_rowconfigure(1, weight=1)
        self.image_frame.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.controls_frame, width=300, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.controls_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas, padding="5")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw", width=280)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        self.canvas.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.scrollbar.pack(side="right", fill="y")

        self.buttons_frame = ttk.Frame(self.image_frame)
        self.buttons_frame.grid(row=0, column=0, sticky="ew", pady=10)

        button_configs = [
            ("Open Folder", self.open_folder, "folder"),
            ("← Previous", self.previous_image, "back"),
            ("Next →", self.next_image, "next"),
            ("Save JPEG", self.save_image, "save"),
            ("Rotate", self.rotate_image, "rotate"),
            ("Flip", self.flip_image_x, "flip")
        ]

        for text, command, name in button_configs:
            btn = ttk.Button(self.buttons_frame, text=text, command=command, width=12)
            btn.pack(side=tk.LEFT, padx=5)
            setattr(self, f"{name}_button", btn)
            if name != "folder":
                btn.state(['disabled'])

        self.image_frame_border = ttk.Frame(self.image_frame, relief="solid", borderwidth=1)
        self.image_frame_border.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        self.image_label = ttk.Label(self.image_frame_border)
        self.image_label.pack(expand=True, fill="both")

        self.create_control_section("Color Adjustments", [
            ("Temperature", "wb_var", -100, 100),
            ("Tint", "tint_var", -100, 100),
            ("Exposure", "exposure_var", -100, 100),
            ("Contrast", "contrast_var", -100, 100)
        ])

        self.create_control_section("Tone Controls", [
            ("Red", "alpha_r_var", -2, 2),
            ("Green", "alpha_g_var", -2, 2),
            ("Blue", "alpha_b_var", -2, 2)
        ])

        self.original_rgb = None
        self.current_image = None
        self.rotation_count = 0
        self.flip_x = False
        self.image_files = []
        self.current_image_index = -1

        self.create_settings_section()
        self.default_settings_path = os.path.join(os.path.dirname(__file__), 'default_settings.yaml')
        
        self.footer_frame = ttk.Frame(root, padding=5)
        self.footer_frame.grid(row=1, column=0, columnspan=2, sticky="ew")

        self.footer_label = ttk.Label(self.footer_frame, text="Version 1.0", anchor="center", font=("Segoe UI", 10))
        self.footer_label.pack(fill="x")
        
    def load_default_settings(self):
        try:
            if os.path.exists(self.default_settings_path):
                with open(self.default_settings_path, 'r') as f:
                    settings = yaml.safe_load(f)
                    self.apply_settings(settings)
                    return True
            return False
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load default settings: {e}")
            return False

    def create_settings_section(self):
        settings_frame = ttk.LabelFrame(self.scrollable_frame, text="Settings", padding=(10, 5, 10, 10))
        settings_frame.pack(fill="x", padx=5, pady=5)

        self.export_button = ttk.Button(settings_frame, text="Export Settings", command=self.export_settings)
        self.export_button.pack(fill="x", pady=2)

        self.import_button = ttk.Button(settings_frame, text="Import Settings", command=self.import_settings)
        self.import_button.pack(fill="x", pady=2)
    
    def get_current_settings(self):
        return {
            'temperature': self.wb_var.get(),
            'tint': self.tint_var.get(),
            'exposure': self.exposure_var.get(),
            'contrast': self.contrast_var.get(),
            'red_tone': self.alpha_r_var.get(),
            'green_tone': self.alpha_g_var.get(),
            'blue_tone': self.alpha_b_var.get(),
            'rotation': self.rotation_count,
            'flip_x': self.flip_x
        }

    def apply_settings(self, settings):
        try:
            self.wb_var.set(settings['temperature'])
            self.tint_var.set(settings['tint'])
            self.exposure_var.set(settings['exposure'])
            self.contrast_var.set(settings['contrast'])
            self.alpha_r_var.set(settings['red_tone'])
            self.alpha_g_var.set(settings['green_tone'])
            self.alpha_b_var.set(settings['blue_tone'])
            self.rotation_count = settings['rotation']
            self.flip_x = settings['flip_x']
            self.update_image()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply settings: {e}")

    def export_settings(self):
        if self.current_image_index < 0:
            messagebox.showinfo("Info", "Please open an image first.")
            return

        save_as_default = messagebox.askyesno(
            "Save Settings",
            "Do you want to save these settings as default settings?\n\n"
            "Yes - Save as default_settings.yaml in the working directory\n"
            "No - Save in the positives directory"
        )

        try:
            settings = self.get_current_settings()

            if save_as_default:
                settings_path = self.default_settings_path
            else:
                current_image_path = self.image_files[self.current_image_index]
                original_dir = os.path.dirname(current_image_path)
                positives_dir = os.path.join(original_dir, 'positives')
                
                if not os.path.exists(positives_dir):
                    os.makedirs(positives_dir)
                
                settings_path = os.path.join(positives_dir, 'image_settings.yaml')

            with open(settings_path, 'w') as f:
                yaml.dump(settings, f, default_flow_style=False)

            messagebox.showinfo("Success", f"Settings exported successfully to:\n{settings_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export settings: {e}")

    def import_settings(self):
        file_path = filedialog.askopenfilename(
            title="Select Settings File",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            with open(file_path, 'r') as f:
                settings = yaml.safe_load(f)
                self.apply_settings(settings)
            messagebox.showinfo("Success", "Settings imported successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to import settings: {e}")
    
    def create_control_section(self, title, controls):
        frame = ttk.LabelFrame(self.scrollable_frame, text=title, padding=(10, 5, 10, 10))
        frame.pack(fill="x", padx=5, pady=5)

        for label, var_name, min_val, max_val in controls:
            control_frame = ttk.Frame(frame)
            control_frame.pack(fill="x", pady=5)
            
            label_widget = ttk.Label(control_frame, text=label)
            label_widget.pack(anchor="w")
            
            setattr(self, var_name, tk.DoubleVar())
            slider = ttk.Scale(control_frame, orient=tk.HORIZONTAL,
                             variable=getattr(self, var_name),
                             from_=min_val, to=max_val)
            slider.pack(fill="x", pady=(2, 0))
            slider.set(0)
            slider.bind("<ButtonRelease-1>", self.update_image)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def open_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.cr2', '.cr3', '.dng'))]
            if self.image_files:
                self.current_image_index = 0
                self.back_button.config(state=tk.NORMAL if self.current_image_index > 0 else tk.DISABLED)
                self.next_button.config(state=tk.NORMAL if self.current_image_index < len(self.image_files) - 1 else tk.DISABLED)
                self.load_image(self.image_files[self.current_image_index])
            else:
                messagebox.showinfo("Info", "No RAW images found in the selected folder.")

    def load_image(self, file_path):
        try:
            rgb_image, r, g, b = process_raw_image(file_path)
            self.original_rgb = (rgb_image, r, g, b)
    
            if self.current_image_index == 0:
                default_settings_applied = self.load_default_settings()
                # Ensure buttons are enabled even if settings are applied
                self.save_button.config(state=tk.NORMAL)
                self.rotate_button.config(state=tk.NORMAL)
                self.flip_button.config(state=tk.NORMAL)
                if default_settings_applied:
                    return 
    
            # Process the image using the default settings
            tone_curve_data, (r_left, r_right, g_left, g_right, b_left, b_right), processed_image = apply_tone_curves(
                rgb_image, r, g, b
            )
    
            # Ensure the updated image is displayed
            self.display_image(processed_image)
    
            # Enable buttons
            self.save_button.config(state=tk.NORMAL)
            self.rotate_button.config(state=tk.NORMAL)
            self.flip_button.config(state=tk.NORMAL)
            self.rotation_count = 0
            self.flip_x = False
    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {e}")



    def previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_image(self.image_files[self.current_image_index])
            self.back_button.config(state=tk.NORMAL if self.current_image_index > 0 else tk.DISABLED)
            self.next_button.config(state=tk.NORMAL)

    def next_image(self):
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_image(self.image_files[self.current_image_index])

            # Apply default settings after loading the image
            self.load_default_settings()

            # Ensure the "Previous" button is enabled when moving forward
            self.back_button.config(state=tk.NORMAL)
            self.next_button.config(state=tk.NORMAL if self.current_image_index < len(self.image_files) - 1 else tk.DISABLED)

            # Update and display the processed image
            self.update_image()

    
    def update_image(self, event=None):
        if self.original_rgb:
            rgb_image, r, g, b = self.original_rgb
            tone_curve_data, (r_left, r_right, g_left, g_right, b_left, b_right), processed_image = apply_tone_curves(
                rgb_image, r, g, b,
                alpha_r=self.alpha_r_var.get(),
                alpha_g=self.alpha_g_var.get(),
                alpha_b=self.alpha_b_var.get()
            )

            # Apply exposure adjustment
            exposure_value = self.exposure_var.get()
            processed_image = adjust_exposure(processed_image, exposure_value)

            # Apply white balance and tint
            wb_value = self.wb_var.get()
            tint_value = self.tint_var.get()
            processed_image = adjust_white_balance(processed_image, wb_value, tint_value)

            # Apply contrast
            contrast_factor = self.contrast_var.get()
            processed_image = adjust_contrast(processed_image, contrast_factor)

            # Apply rotation if any
            if self.rotation_count > 0:
                processed_image = processed_image.rotate(90 * self.rotation_count, expand=True)

            # Apply flip X if enabled
            if self.flip_x:
                processed_image = processed_image.transpose(Image.FLIP_LEFT_RIGHT)

            self.display_image(processed_image)

    def display_image(self, image):
        # Calculate scaling factor to fit the image in the available space
        frame_width = self.image_frame.winfo_width()
        frame_height = self.image_frame.winfo_height() - self.buttons_frame.winfo_height()

        # Get image size
        img_width, img_height = image.size

        # Calculate scaling factor
        width_ratio = frame_width / img_width
        height_ratio = frame_height / img_height
        scale_factor = min(width_ratio, height_ratio)

        # Calculate new size
        new_width = int(img_width * scale_factor)
        new_height = int(img_height * scale_factor)

        # Resize image
        image_resized = image.resize((new_width, new_height), Image.LANCZOS)
        photo = ImageTk.PhotoImage(image_resized)

        self.image_label.configure(image=photo)
        self.image_label.image = photo
        self.current_image = image
        

    def save_image(self):
        if self.current_image and self.current_image_index >= 0:
            # Get the original file path
            original_file_path = self.image_files[self.current_image_index]
            original_dir = os.path.dirname(original_file_path)

            # Create 'positives' directory if it doesn't exist
            positives_dir = os.path.join(original_dir, 'positives')
            if not os.path.exists(positives_dir):
                os.makedirs(positives_dir)

            # Get original filename without extension and create new filename
            original_filename = os.path.splitext(os.path.basename(original_file_path))[0]
            new_filename = f"{original_filename}_processed.jpg"

            # Create full save path
            save_path = os.path.join(positives_dir, new_filename)

            try:
                self.current_image.save(save_path, "JPEG", quality=95)
                messagebox.showinfo("Success", f"Image saved successfully as:\n{save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {e}")

    def rotate_image(self):
        if self.current_image:
            self.rotation_count = (self.rotation_count + 1) % 4
            self.update_image()

    def flip_image_x(self):
        if self.current_image:
            self.flip_x = not self.flip_x  # Toggle flip state
            self.update_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = RawImageProcessorApp(root)

    # Start profiling
    profiler = cProfile.Profile()
    profiler.enable()

    # Run the main loop
    root.mainloop()

    # Stop profiling
    profiler.disable()

    # Print profiling results to the terminal
    stats = pstats.Stats(profiler)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()