import sys
import os
import rawpy
import numpy as np
from skimage.exposure import match_histograms, equalize_hist, equalize_adapthist
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QHBoxLayout, QSlider
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

class NegativeImageProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.image = self._load_raw_image()
        self.original_image = self.image.copy()  # Store the original image for reference

    def _load_raw_image(self):
        """
        Loads a raw image file (CR2, CR3, or DNG) and converts it to an image array.

        Returns:
            ndarray: Loaded image array.
        """
        with rawpy.imread(self.file_path) as raw:
            return raw.postprocess()
    
    def process_raw_image_with_equalize(self):
        """
        Process the raw image file:
        1. Perform histogram matching to align RGB channels.
        2. Invert the colors.
        3. Apply histogram equalization.

        Returns:
            ndarray: Processed image.
        """
        # Step 1: Histogram matching
        matched_img = self.image.copy()
        matched_img[:, :, 0] = match_histograms(self.image[:, :, 0], self.image[:, :, 1])  # Match red to green
        matched_img[:, :, 2] = match_histograms(self.image[:, :, 2], self.image[:, :, 1])  # Match blue to green

        # Step 2: Invert the colors
        inverted_img = 255 - matched_img
        r, g, b = inverted_img[:,:,0], inverted_img[:,:,1], inverted_img[:,:,2] 
        # Apply adaptive histogram equalization to each channel

        r_eq = equalize_adapthist(r)

        g_eq = equalize_adapthist(g)

        b_eq = equalize_adapthist(b)

        # Step 3: Apply histogram equalization
        #equalized_img = equalize_hist(inverted_img)
        equalized_img = np.stack([r_eq, g_eq, b_eq], axis=2)
        return equalized_img

    def adjust_image_exposure(self, image, exposure_value=0):
        exposure_value = max(-3, min(exposure_value, 3))
        scaling_factor = 2 ** exposure_value
        adjusted_image = image * scaling_factor
        return np.clip(adjusted_image, 0, 1)

    def adjust_image_contrast(self, image, contrast_factor=0.0):
        contrast_factor = max(-1, min(contrast_factor, 1))
        mean_intensity = np.mean(image)
        adjusted_image = (image - mean_intensity) * (1 + contrast_factor) + mean_intensity
        return np.clip(adjusted_image, 0, 1)

    def adjust_gamma(self, image, gamma=1.0):
        if gamma <= 0:
            raise ValueError("Gamma value must be greater than 0.")
        adjusted_image = np.power(image, gamma)
        return np.clip(adjusted_image, 0, 1)

    def adjust_white_balance(self, image, wb_factor=0.0):
        if image.shape[-1] != 3:
            raise ValueError("Input image must have 3 color channels (RGB).")
        wb_factor *= 0.5
        red_gain = 1 + wb_factor
        blue_gain = 1 - wb_factor
        green_gain = 1
        adjusted_image = image.copy()
        adjusted_image[:, :, 0] *= red_gain
        adjusted_image[:, :, 1] *= green_gain
        adjusted_image[:, :, 2] *= blue_gain
        return np.clip(adjusted_image, 0, 1)

    def adjust_image_tint(self, image, tint_factor=0.0):
        if image.shape[-1] != 3:
            raise ValueError("Input image must have 3 color channels (RGB).")
        tint_factor *= 0.5
        green_gain = 1 - tint_factor
        red_blue_gain = 1 + tint_factor
        adjusted_image = image.copy()
        adjusted_image[:, :, 0] *= red_blue_gain
        adjusted_image[:, :, 1] *= green_gain
        adjusted_image[:, :, 2] *= red_blue_gain
        return np.clip(adjusted_image, 0, 1)

    def adjust_vibrance(self, image, vibrance_factor=0.0):
        vibrance_factor = max(-1, min(vibrance_factor, 1))
        luminance = np.mean(image, axis=2, keepdims=True)
        adjusted_image = image + (image - luminance) * vibrance_factor
        return np.clip(adjusted_image, 0, 1)

    def adjust_saturation(self, image, saturation_factor=0.0):
        saturation_factor = max(-1, min(saturation_factor, 1))
        luminance = np.mean(image, axis=2, keepdims=True)
        adjusted_image = luminance + (image - luminance) * (1 + saturation_factor)
        return np.clip(adjusted_image, 0, 1)

    def adjust_red_channel(self, image, red_factor=0.0):
        red_factor = max(-1, min(red_factor, 1))
        adjusted_image = image.copy()
        adjusted_image[:, :, 0] *= (1 + red_factor)
        return np.clip(adjusted_image, 0, 1)

    def adjust_green_channel(self, image, green_factor=0.0):
        green_factor = max(-1, min(green_factor, 1))
        adjusted_image = image.copy()
        adjusted_image[:, :, 1] *= (1 + green_factor)
        return np.clip(adjusted_image, 0, 1)

    def adjust_blue_channel(self, image, blue_factor=0.0):
        blue_factor = max(-1, min(blue_factor, 1))
        adjusted_image = image.copy()
        adjusted_image[:, :, 2] *= (1 + blue_factor)
        return np.clip(adjusted_image, 0, 1)

    def flip_image_lr(self, image):
        return np.fliplr(image)

    def rotate_image_90(self, image):
        return np.rot90(image)

    def apply_vignetting(self, image, strength=0.5):
        strength = max(0, min(strength, 1))
        rows, cols = image.shape[:2]
        y, x = np.ogrid[:rows, :cols]
        center_x, center_y = cols / 2, rows / 2
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt((2 * center_x)**2 + (2 * center_y)**2)
        radial_gradient = 1 - (distance / max_distance)
        radial_gradient = np.clip(radial_gradient, 0, 1) ** (1 / (1 - strength))
        adjusted_image = image.copy()
        for i in range(3):
            adjusted_image[:, :, i] *= radial_gradient
        return np.clip(adjusted_image, 0, 1)


class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.slider_values = {}  # Track current slider values
        self.slider_labels = {}  # Initialize the slider_labels dictionary
        self.sliders = {}  # Initialize the sliders dictionary
        self.image_slider_states = {}  # Store slider states for each image
        self.initUI()
        self.image_files = []
        self.current_image_index = -1
        self.processed_image = None
        self.original_processed_image = None  # Store the original processed image
        self.processor = None

    def initUI(self):
        self.setWindowTitle('Negative Image Processor')
        self.setGeometry(100, 100, 1200, 800)

        # Use a better font
        font = self.font()
        font.setPointSize(10)
        self.setFont(font)

        # Central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Main layout: horizontal split between controls (left) and image (right)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Left panel for controls
        self.controls_panel = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_panel)
        self.main_layout.addWidget(self.controls_panel, stretch=1)  # Controls take 1/3 of the width

        # Right panel for image display
        self.image_panel = QWidget()
        self.image_layout = QVBoxLayout(self.image_panel)
        self.main_layout.addWidget(self.image_panel, stretch=2)  # Image takes 2/3 of the width

        # Image display
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(600, 600)  # Set a minimum size for the QLabel
        self.image_layout.addWidget(self.image_label)

        # Buttons for navigation and actions
        self.button_layout = QVBoxLayout()

        self.open_dir_button = QPushButton('Open Directory', self)
        self.open_dir_button.clicked.connect(self.open_directory)
        self.button_layout.addWidget(self.open_dir_button)

        self.prev_button = QPushButton('Previous Image', self)
        self.prev_button.clicked.connect(self.show_previous_image)
        self.button_layout.addWidget(self.prev_button)

        self.next_button = QPushButton('Next Image', self)
        self.next_button.clicked.connect(self.show_next_image)
        self.button_layout.addWidget(self.next_button)

        self.rotate_button = QPushButton('Rotate 90°', self)
        self.rotate_button.clicked.connect(self.rotate_image)
        self.button_layout.addWidget(self.rotate_button)

        self.flip_button = QPushButton('Flip Horizontal', self)
        self.flip_button.clicked.connect(self.flip_image)
        self.button_layout.addWidget(self.flip_button)

        self.save_button = QPushButton('Save Image', self)
        self.save_button.clicked.connect(self.save_image)
        self.button_layout.addWidget(self.save_button)

        self.apply_button = QPushButton('Apply Settings', self)
        self.apply_button.clicked.connect(self.apply_settings)
        self.button_layout.addWidget(self.apply_button)

        self.reset_button = QPushButton('Reset Edits', self)
        self.reset_button.clicked.connect(self.reset_edits)
        self.button_layout.addWidget(self.reset_button)

        self.controls_layout.addLayout(self.button_layout)

        # Sliders for adjustments
        adjustments = [
            ("Exposure", -3, 3),
            ("Contrast", -1, 1),
            ("Gamma", 0.1, 3),
            ("White Balance", -1, 1),
            ("Tint", -1, 1),
            ("Vibrance", -1, 1),
            ("Saturation", -1, 1),
            ("Red", -1, 1),
            ("Green", -1, 1),
            ("Blue", -1, 1),
            ("Vignetting", 0, 1),
        ]

        for name, min_val, max_val in adjustments:
            slider_layout = QHBoxLayout()
            label = QLabel(f"{name}:")
            slider_layout.addWidget(label)

            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(int(min_val * 100))
            slider.setMaximum(int(max_val * 100))
            slider.setValue(100 if name == "Gamma" else 0)  # Set Gamma default to 1 (100 in scaled range)
            slider.valueChanged.connect(lambda value, n=name: self.update_slider_value(n, value))
            slider_layout.addWidget(slider)

            value_label = QLabel("0.0")
            slider_layout.addWidget(value_label)

            self.sliders[name] = slider
            self.slider_labels[name] = value_label
            self.slider_values[name] = slider.value() / 100  # Store initial slider value
            self.controls_layout.addLayout(slider_layout)

        # Add stretch to push controls to the top
        self.controls_layout.addStretch()

    def update_slider_value(self, name, value):
        """Update the slider value and its display label."""
        scaled_value = value / 100
        self.slider_values[name] = scaled_value
        self.slider_labels[name].setText(f"{scaled_value:.2f}")

    def open_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Open Directory")
        if directory:
            self.image_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.cr2', '.cr3', '.dng'))]
            if self.image_files:
                self.current_image_index = 0
                self.load_image(self.image_files[self.current_image_index])

    def load_image(self, file_path):
        # Save current slider states for the current image (if any)
        if self.current_image_index >= 0 and self.image_files:
            self.image_slider_states[self.image_files[self.current_image_index]] = {
                name: slider.value() for name, slider in self.sliders.items()
            }

        # Load the new image
        self.processor = NegativeImageProcessor(file_path)
        self.processed_image = self.processor.process_raw_image_with_equalize()
        self.original_processed_image = self.processed_image.copy()  # Store the original processed image

        # Restore slider states if the image has been opened before
        if file_path in self.image_slider_states:
            for name, value in self.image_slider_states[file_path].items():
                self.sliders[name].setValue(value)
                self.update_slider_value(name, value)
        else:
            # Reset sliders to default values for new images
            for name, slider in self.sliders.items():
                slider.setValue(100 if name == "Gamma" else 0)
                self.update_slider_value(name, slider.value())

        self.display_image(self.processed_image)

    def display_image(self, image):
        # Scale the image to the range 0-255 and convert to uint8
        scaled_image = (image * 255).astype(np.uint8)

        # Ensure the image is in RGB format (height, width, 3)
        if scaled_image.ndim == 3 and scaled_image.shape[2] == 3:
            height, width, channel = scaled_image.shape
            bytes_per_line = 3 * width

            # Convert the numpy array to a bytes object
            image_data = scaled_image.tobytes()

            # Create QImage from the bytes object
            q_img = QImage(image_data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Scale the image to fit the QLabel while maintaining aspect ratio
            pixmap = QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(
                self.image_label.width(), self.image_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(pixmap)
        else:
            print("Error: Image is not in the expected RGB format.")

    def show_previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_image(self.image_files[self.current_image_index])

    def show_next_image(self):
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_image(self.image_files[self.current_image_index])

    def rotate_image(self):
        if self.processor and self.processed_image is not None:
            self.processed_image = self.processor.rotate_image_90(self.processed_image)
            self.display_image(self.processed_image)

    def flip_image(self):
        if self.processor and self.processed_image is not None:
            self.processed_image = self.processor.flip_image_lr(self.processed_image)
            self.display_image(self.processed_image)

    def apply_settings(self):
        if self.processor and self.original_processed_image is not None:
            # Apply adjustments only for sliders that have changed
            adjusted_image = self.original_processed_image.copy()
            for name, value in self.slider_values.items():
                if name == "Exposure":
                    adjusted_image = self.processor.adjust_image_exposure(adjusted_image, value)
                elif name == "Contrast":
                    adjusted_image = self.processor.adjust_image_contrast(adjusted_image, value)
                elif name == "Gamma":
                    adjusted_image = self.processor.adjust_gamma(adjusted_image, value)
                elif name == "White Balance":
                    adjusted_image = self.processor.adjust_white_balance(adjusted_image, value)
                elif name == "Tint":
                    adjusted_image = self.processor.adjust_image_tint(adjusted_image, value)
                elif name == "Vibrance":
                    adjusted_image = self.processor.adjust_vibrance(adjusted_image, value)
                elif name == "Saturation":
                    adjusted_image = self.processor.adjust_saturation(adjusted_image, value)
                elif name == "Red":
                    adjusted_image = self.processor.adjust_red_channel(adjusted_image, value)
                elif name == "Green":
                    adjusted_image = self.processor.adjust_green_channel(adjusted_image, value)
                elif name == "Blue":
                    adjusted_image = self.processor.adjust_blue_channel(adjusted_image, value)
                elif name == "Vignetting":
                    adjusted_image = self.processor.apply_vignetting(adjusted_image, value)

            # Update the processed_image and display it
            self.processed_image = adjusted_image
            self.display_image(self.processed_image)

    def reset_edits(self):
        """Reset the image to its original state and reset sliders to default values."""
        if self.original_processed_image is not None:
            self.processed_image = self.original_processed_image.copy()
            self.display_image(self.processed_image)

            # Reset sliders to default values
            for name, slider in self.sliders.items():
                slider.setValue(100 if name == "Gamma" else 0)
                self.update_slider_value(name, slider.value())

    def save_image(self):
        if self.processed_image is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png);;JPEG Files (*.jpg)")
            if file_path:
                from skimage.io import imsave
                imsave(file_path, (self.processed_image * 255).astype(np.uint8))
                
if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())