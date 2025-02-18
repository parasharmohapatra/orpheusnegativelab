'''
AUTHOR: Parashar Mohapatra
'''

import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QSlider, QFileDialog, 
                            QScrollArea, QGroupBox, QStatusBar, QFrame, QSizePolicy, QGridLayout, QProgressBar)
from PyQt5.QtCore import Qt, QSize, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont, QFontDatabase, QTransform
import numpy as np
from PIL import Image
from neg_processor import NumbaOptimizedNegativeImageProcessor, ImageFileManager
import cProfile
import pstats
from PyQt5 import uic

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
        super().__init__()
        
        # Load the UI file
        uic.loadUi('app.ui', self)
        
        # Initialize status bar and progress bar
        self.status_bar = self.statusbar
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(15)
        self.progress_bar.hide()
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        # Initialize update timer
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.process_image_update)
        
        # Disable navigation buttons initially
        #self.prevButton.setEnabled(False)
        #self.nextButton.setEnabled(False)
        
        # Initialize variables
        self.processor = None
        self.image_index = 0
        self.current_image = None
        self.directory_path = ""
        self.loaded_images = {}
        self.current_image_path = None
        self.raw_file_paths = []
        self.image_saver_thread = None
        self.image_loader_thread = None
        self.processed_pixmap = None
        self.rotation_angle = 0
        self.flip_horizontal = False

        # Connect signals
        self.openDirButton.clicked.connect(self.open_directory)
        self.rotateButton.clicked.connect(self.rotate_image)
        self.flipButton.clicked.connect(self.flip_image)
        self.resetButton.clicked.connect(self.reset_sliders)
        self.prevButton.clicked.connect(self.prev_image)
        self.nextButton.clicked.connect(self.next_image)

        # Initialize sliders
        self.setup_sliders()

        # Show window
        self.showMaximized()

    def setup_sliders(self):
        # Configure sliders
        slider_configs = {
            'tint': (90, 110, 100),
            'whiteBalance': (90, 110, 100),
            'blacks': (-100, 100, 0),
            'whites': (-100, 100, 0),
            'highlights': (-100, 100, 0),
            'shadows': (-100, 100, 0),
            'gamma': (0, 100, 50),
            'log': (100, 200, 150),
        }

        for name, (min_val, max_val, default) in slider_configs.items():
            slider = getattr(self, f"{name}Slider")
            value_label = getattr(self, f"{name}Value")
            
            slider.setMinimum(min_val)
            slider.setMaximum(max_val)
            slider.setValue(default)
            
            # Create a non-lambda function to avoid closure issues
            def create_value_updater(label):
                return lambda val: label.setText(f"{val/100:.2f}")
            
            value_updater = create_value_updater(value_label)
            slider.valueChanged.connect(value_updater)
            slider.sliderReleased.connect(self.trigger_update)
            
            # Trigger initial value update
            value_updater(default)

    def get_slider_values(self):
        return {
            'tint_adj_factor': self.tintSlider.value() / 100,  # Updated to tint
            'white_balance_adj_factor': self.whiteBalanceSlider.value() / 100,  # Updated to white balance
            'blacks': self.blacksSlider.value(),
            'whites': self.whitesSlider.value(),
            'highlights': self.highlightsSlider.value(),
            'shadows': self.shadowsSlider.value(),
            'gamma_adj': self.gammaSlider.value() / 100,
            'log_adj': self.logSlider.value() / 100
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
                # Disable navigation buttons while loading
                #self.prevButton.setEnabled(False)
                #self.nextButton.setEnabled(False)
                
                self.file_manager = ImageFileManager(self.directory_path)
                self.image_processor = NumbaOptimizedNegativeImageProcessor()
                self.image_index = 0
                self.loaded_images = {}
                self.raw_file_paths = []

                for file_path in self.file_manager.file_paths:
                    if file_path.lower().endswith(('.cr2', '.cr3', '.crw', '.raw', '.nef', '.arw', '.nrw', '.rw2', '.srf', '.sr2', '.dng')): # Check file extension
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
        self.progress_bar.hide()
        self.status_bar.showMessage("Loading complete")
        
        # Enable navigation buttons if there are images
        if len(self.raw_file_paths) > 0:
            self.nextButton.setEnabled(len(self.raw_file_paths) > 1)
            self.prevButton.setEnabled(True)  # Start at first image, so prev is disabled

    def prev_image(self):
        try:
            self.save_current_image()
            if self.file_manager and self.image_index > 0:
                self.image_index -= 1
                file_path = self.raw_file_paths[self.image_index]
                
                # Check if image is loaded, if not try to load it
                if file_path not in self.loaded_images:
                    try:
                        image_data = self.file_manager.load_image(file_path)
                        self.loaded_images[file_path] = image_data
                    except Exception as e:
                        self.status_bar.showMessage(f"Error loading image {os.path.basename(file_path)}: {str(e)}")
                        # Skip to previous image
                        self.prev_image()
                        return
                
                self.image_processor.open_image(self.loaded_images[file_path])
                self.current_image_path = file_path
                self.display_image()
                self.status_bar.showMessage(f"Showing image {self.image_index + 1} of {len(self.raw_file_paths)}")
        except Exception as e:
            self.status_bar.showMessage(f"Error navigating to previous image: {str(e)}")

    def next_image(self):
        try:
            self.save_current_image()
            if self.file_manager and self.image_index < len(self.raw_file_paths) - 1:
                self.image_index += 1
                file_path = self.raw_file_paths[self.image_index]
                
                # Check if image is loaded, if not try to load it
                if file_path not in self.loaded_images:
                    try:
                        image_data = self.file_manager.load_image(file_path)
                        self.loaded_images[file_path] = image_data
                    except Exception as e:
                        self.status_bar.showMessage(f"Error loading image {os.path.basename(file_path)}: {str(e)}")
                        # Skip to next image
                        self.next_image()
                        return
                
                self.image_processor.open_image(self.loaded_images[file_path])
                self.current_image_path = file_path
                self.display_image()
                self.status_bar.showMessage(f"Showing image {self.image_index + 1} of {len(self.raw_file_paths)}")
        except Exception as e:
            self.status_bar.showMessage(f"Error navigating to next image: {str(e)}")

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
            # Get current adjustments
            adjustments = self.get_slider_values()
            
            # Process the image with current adjustments
            processed_array = self.image_processor.process_image(**adjustments)
            
            # Convert to QImage
            height, width, channel = processed_array.shape
            bytes_per_line = 3 * width
            
            # Convert to uint8 and ensure proper scaling
            processed_array = np.clip(processed_array / 256, 0, 255).astype(np.uint8)
            
            img_qt = QImage(processed_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            if img_qt.isNull():
                self.status_bar.showMessage("Failed to create QImage")
                self.image_label.clear()
                self.processed_pixmap = None  # Reset pixmap
                return

            # Create and store the processed pixmap
            self.processed_pixmap = QPixmap.fromImage(img_qt)
            
            # Apply transformations and display
            self.transform_and_display()
            
            # Update status
            self.status_bar.showMessage(f"Displaying image {self.image_index + 1} of {len(self.raw_file_paths)}")

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
            'tint': (90, 110, 100),
            'whiteBalance': (90, 110, 100),
            'blacks': (-100, 100, 0),
            'whites': (-100, 100, 0),
            'highlights': (-100, 100, 0),
            'shadows': (-100, 100, 0),
            'gamma': (0, 100, 50),
            'log': (100, 200, 150),
        }

        for name, (min_val, max_val, default) in slider_configs.items():
            slider = getattr(self, f"{name}Slider")
            slider.setValue(default)

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
