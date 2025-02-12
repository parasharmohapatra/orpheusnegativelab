'''
AUTHOR: Parashar Mohapatra

MIT License

Copyright (c) 2025 Parashar Mohapatra

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QSlider, QFileDialog, 
                            QScrollArea, QGroupBox, QStatusBar, QFrame, QSizePolicy, QGridLayout, QProgressBar)
from PyQt5.QtCore import Qt, QSize, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont, QFontDatabase, QTransform
from PyQt5 import uic
import numpy as np
from PIL import Image
from neg_processor import NumbaOptimizedNegativeImageProcessor, ImageFileManager
import cProfile
import pstats

class ImageLoaderThread(QThread):  # Thread for loading images
    loaded_image = pyqtSignal(str, np.ndarray)  # Signal for each loaded image
    finished_loading = pyqtSignal()
    progress_update = pyqtSignal(int)

    def __init__(self, file_paths, file_manager):
        super().__init__()
        self.file_paths = file_paths
        self.file_manager = file_manager
        self.loaded_images = {}

    def run(self):
        total_files = len(self.file_paths)
        for i, file_path in enumerate(self.file_paths):
            try:
                image_data = self.file_manager.load_image(file_path)
                self.loaded_image.emit(file_path, image_data)  # Emit signal with data
            except Exception as e:
                print(f"Skipping invalid file: {file_path} - {e}")
            self.progress_update.emit(int((i + 1) / total_files * 100)) # Emit progress
        self.finished_loading.emit()

class ImageSaverThread(QThread):
    finished_saving = pyqtSignal(str)
    error_saving = pyqtSignal(str)

    def __init__(self, image_data, save_path, rotation_angle, flip_horizontal):  # Add flip_horizontal
        super().__init__()
        self.image_data = image_data
        self.save_path = save_path
        self.rotation_angle = rotation_angle
        self.flip_horizontal = flip_horizontal  # Store flip setting

    def run(self):
        try:
            # *** PREPARE IMAGE FOR SAVING (INCLUDING ROTATION AND FLIP) ***
            processed_image = np.clip(self.image_data / 256, 0, 255).astype(np.uint8)

            image = Image.fromarray(processed_image)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            rotated_image = image.rotate(-self.rotation_angle, expand=True)  # Rotate

            if self.flip_horizontal:  # Apply flip if needed
                flipped_image = rotated_image.transpose(Image.FLIP_LEFT_RIGHT) # PIL flip
            else:
                flipped_image = rotated_image

            flipped_image.save(self.save_path, "JPEG", quality=100)
            self.finished_saving.emit(self.save_path)
        except Exception as e:
            self.error_saving.emit(str(e))

class ModernNegativeImageGUI(QMainWindow):
    def __init__(self):
        # Load the UI file
        super().__init__()
        uic.loadUi('orpheus.ui', self)

        # Initialize variables
        self.processor = None
        self.image_index = 0
        self.current_image = None
        self.directory_path = ""
        self.loaded_images = {}
        self.current_image_path = None
        self.raw_file_paths = []  # List to store only raw file paths
        self.image_saver_thread = None  # Add the saver thread attribute
        self.image_loader_thread = None  # Add thread attribute
        self.processed_pixmap = None  # Store the processed pixmap
        self.rotation_angle = 0
        self.flip_horizontal = False  # Initialize flip state

        # Initialize debounce timer
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.process_image_update)

        # Connect signals
        self.open_button.clicked.connect(self.open_directory)
        self.rotate_button.clicked.connect(self.rotate_image)
        self.flip_button.clicked.connect(self.flip_image)
        self.reset_button.clicked.connect(self.reset_sliders)
        self.prev_button.clicked.connect(self.prev_image)
        self.next_button.clicked.connect(self.next_image)

        # Connect slider signals and store sliders
        self.sliders = {}
        slider_names = ['r', 'g', 'b', 'exposure', 'gamma']
        for name in slider_names:
            slider = getattr(self, f'{name}_slider')
            value_label = getattr(self, f'{name}_value')
            slider.valueChanged.connect(
                lambda value, lbl=value_label: lbl.setText(f"{value / 100:.2f}")
            )
            slider.sliderReleased.connect(self.trigger_update)
            self.sliders[name.upper() if name in ['r', 'g', 'b'] else name.capitalize()] = (slider, value_label)

        # Set window size
        self.showMaximized()

    def get_slider_values(self):
        return {
            'r_adjust': self.sliders['R'][0].value() / 100,
            'g_adjust': self.sliders['G'][0].value() / 100,
            'b_adjust': self.sliders['B'][0].value() / 100,
            'exposure': self.sliders['Exposure'][0].value() / 100,
            'gamma': self.sliders['Gamma'][0].value() / 100
        }

    def update_slider_label(self, name):
        pass

    def trigger_update(self):
        self.update_timer.start(100)

    def process_image_update(self):
        self.display_image()

    def open_directory(self):
        self.directory_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if self.directory_path:
            try:
                self.file_manager = ImageFileManager(self.directory_path)
                self.image_processor = NumbaOptimizedNegativeImageProcessor()
                self.image_index = 0
                self.loaded_images = {}
                self.raw_file_paths = []  # Clear previous raw file paths

                for file_path in self.file_manager.file_paths:
                    if file_path.lower().endswith(('.cr2', '.cr3', '.raw', '.nef')): # Check file extension
                        self.raw_file_paths.append(file_path) # Add only raw paths
                
                self.progress_bar.show()
                self.progress_bar.setValue(0)
                self.status_bar.showMessage("Loading images...")

                self.image_loader_thread = ImageLoaderThread(self.raw_file_paths, self.file_manager) # Use only raw file paths
                self.image_loader_thread.loaded_image.connect(self.add_loaded_image)
                self.image_loader_thread.finished_loading.connect(self.loading_finished)
                self.image_loader_thread.progress_update.connect(self.update_progress)
                self.image_loader_thread.start()

            except Exception as e:
                self.status_bar.showMessage(f"Error loading directory: {str(e)}")
                self.image_label.clear()
                self.file_manager = None
                self.image_processor = None
                self.progress_bar.hide()
                return
    
    def add_loaded_image(self, file_path, image_data): # Slot to add images as they are loaded
        self.loaded_images[file_path] = image_data
        if not self.current_image_path: # if this is the first image loaded
            self.image_processor.open_image(image_data)
            self.current_image_path = file_path
            self.display_image()

    def update_progress(self, progress): # slot to update the progress bar
        self.progress_bar.setValue(progress)

    def loading_finished(self):
        self.progress_bar.hide()  # Hide progress bar
        self.status_bar.showMessage(f"Loaded directory: {self.directory_path}")
        if not self.loaded_images:  # No valid images found
            self.status_bar.showMessage("No valid image files found in the directory.")
            self.image_label.clear()
            self.file_manager = None
            self.image_processor = None
            return

    def prev_image(self):
        self.save_current_image()
        if self.file_manager and self.image_index > 0:
            self.image_index -= 1
            file_path = self.raw_file_paths[self.image_index] # Use raw_file_paths
            self.image_processor.open_image(self.loaded_images[file_path])
            self.current_image_path = file_path
            self.display_image()
            self.status_bar.showMessage(f"Showing image {self.image_index + 1} of {len(self.raw_file_paths)}")  # Use raw_file_paths

    def next_image(self):
        self.save_current_image()
        if self.file_manager and self.image_index < len(self.raw_file_paths) - 1:  # Use raw_file_paths
            self.image_index += 1
            file_path = self.raw_file_paths[self.image_index]  # Use raw_file_paths
            self.image_processor.open_image(self.loaded_images[file_path])
            self.current_image_path = file_path
            self.display_image()
            self.status_bar.showMessage(f"Showing image {self.image_index + 1} of {len(self.raw_file_paths)}")  # Use raw_file_paths

    def rotate_image(self):
        self.rotation_angle = (self.rotation_angle - 90) % 360
        self.transform_and_display()  # Call transform_and_display

    def flip_image(self):
        self.flip_horizontal = not self.flip_horizontal
        self.transform_and_display()  # Call transform_and_display

    def transform_and_display(self):  # New function for transformation
        if self.processed_pixmap is None:  # Check if pixmap exists
            return

        transform = QTransform()
        transform.rotate(self.rotation_angle)
        if self.flip_horizontal:
            transform.scale(-1, 1)

        rotated_pixmap = self.processed_pixmap.transformed(transform, Qt.TransformationMode.SmoothTransformation)  # Transform existing pixmap

        label_size = self.image_label.size()
        if label_size.width() <= 0 or label_size.height() <= 0:
            return

        scaled_pixmap = rotated_pixmap.scaled(
            label_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.image_label.setPixmap(scaled_pixmap)

    def display_image(self):
        if not self.file_manager or not self.image_processor or self.image_processor.current_rgb is None:
            self.image_label.clear()
            self.processed_pixmap = None  # Reset pixmap
            return

        try:
            values = self.get_slider_values()
            processed_image = self.image_processor.process_image(
                values['r_adjust'], values['g_adjust'], values['b_adjust'],
                values['gamma'], values['exposure']
            )

            if processed_image is None or processed_image.size == 0:
                self.status_bar.showMessage("Invalid image data")
                self.image_label.clear()
                self.processed_pixmap = None  # Reset pixmap
                return

            processed_image = np.clip(processed_image / 256, 0, 255).astype(np.uint8)

            height, width, channels = processed_image.shape
            bytesPerLine = processed_image.strides[0]

            if channels == 3:
                format = QImage.Format_RGB888
            elif channels == 4:
                format = QImage.Format_RGBA8888
            else:
                self.status_bar.showMessage("Unsupported image format")
                self.image_label.clear()
                self.processed_pixmap = None  # Reset pixmap
                return

            img_qt = QImage(processed_image.data, width, height, bytesPerLine, format)

            if img_qt.isNull():
                self.status_bar.showMessage("Failed to create QImage")
                self.image_label.clear()
                self.processed_pixmap = None  # Reset pixmap
                return

            pixmap = QPixmap.fromImage(img_qt)

            if pixmap.isNull():
                self.status_bar.showMessage("Failed to create QPixmap")
                self.image_label.clear()
                self.processed_pixmap = None  # Reset pixmap
                return

            self.processed_pixmap = pixmap  # Store the pixmap

            self.transform_and_display()  # Call transform_and_display to apply initial transform and display

        except Exception as e:
            self.status_bar.showMessage(f"Error displaying image: {str(e)}")
            print(f"Display Error: {e}")
            self.image_label.clear()
            self.processed_pixmap = None  # Reset pixmap

    def save_current_image(self):
        if not self.file_manager or not self.image_processor or self.image_processor.current_rgb is None or self.current_image_path is None:
            return

        try:
            positives_dir = os.path.join(self.directory_path, "positives")
            os.makedirs(positives_dir, exist_ok=True)

            file_name = os.path.basename(self.current_image_path).lower()
            for ext in [".cr2", ".cr3", ".raw", ".nef"]:
                file_name = file_name.replace(ext, ".jpg")

            save_path = os.path.join(positives_dir, file_name)

            self.image_saver_thread = ImageSaverThread(self.image_processor.current_rgb, save_path, self.rotation_angle, self.flip_horizontal)  # Pass flip setting
            self.image_saver_thread.finished_saving.connect(self.saving_finished)
            self.image_saver_thread.error_saving.connect(self.saving_error)
            self.image_saver_thread.start()
            self.status_bar.showMessage(f"Saving image: {file_name}...")

        except Exception as e:
            self.status_bar.showMessage(f"Error preparing to save: {str(e)}")

    def saving_finished(self, save_path):
        file_name = os.path.basename(save_path)
        self.status_bar.showMessage(f"Saved image: {file_name}")

    def saving_error(self, error_message):
        self.status_bar.showMessage(f"Error saving image: {error_message}")

    def reset_sliders(self):
        # Reset all sliders to their default values
        slider_configs = {  # Get your slider configurations
            'R': (-0.5, 0.5, 0),
            'G': (-0.5, 0.5, -0.15),
            'B': (-0.5, 0.5, -0.3),
            'Exposure': (-1, 1, 0),
            'Gamma': (0.1, 4, 1)
        }

        for name, (min_val, max_val, default) in slider_configs.items():
            slider, label = self.sliders[name]
            slider.setValue(int(default * 100))  # Set slider to default value
            label.setText(f"{default:.2f}")  # Update the label

        self.display_image() # Refresh the image to reflect the reset sliders


def main():
    app = QApplication(sys.argv)
    font = QFont('Roboto')
    app.setFont(font)

    if app.font().family() != 'Roboto':
        print("Roboto not found, using a fallback font.")
        font = QFont('Arial')
        app.setFont(font)
    else:
        app.setFont(font)
    
    app.setStyleSheet(f"""
        * {{
            color: #333333;
        }}
    """)
    
    window = ModernNegativeImageGUI()
    window.show()

    # Profiling setup
    profiler = cProfile.Profile()
    profiler.enable()  # Start profiling

    app.exec() # Profile during app execution

    profiler.disable()  # Stop profiling

    # Save profile stats
    profile_file = "app_profile.pstat"
    profiler.dump_stats(profile_file)

    print(f"Profiling data saved to {profile_file}")




if __name__ == "__main__":
    main()
