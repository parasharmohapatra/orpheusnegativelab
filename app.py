'''
AUTHOR: Parashar Mohapatra
'''

import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QSlider, QFileDialog,
                             QScrollArea, QGroupBox, QStatusBar, QFrame, QSizePolicy, QGridLayout, QProgressBar,
                             QRubberBand)
from PyQt5.QtCore import Qt, QSize, QTimer, QThread, pyqtSignal, QRect, QPoint
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont, QFontDatabase, QTransform, QCursor
import numpy as np
from PIL import Image
from neg_processor import NegativeImageProcessor, ImageFileManager
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

    def __init__(self, image_data, save_path, rotation_angle, flip_horizontal, is_preprocessed=False):
        super().__init__()
        self.image_data = image_data
        self.save_path = save_path
        self.rotation_angle = rotation_angle
        self.flip_horizontal = flip_horizontal
        self.is_preprocessed = is_preprocessed  # Flag to indicate if image is already processed

    def run(self):
        try:
            # Process image data based on whether it's already preprocessed
            if self.is_preprocessed:
                # For cropped images that are already in the [0, 255] range
                processed_image = self.image_data
            else:
                # For original images that need scaling
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
        self.is_cropping = False
        self.crop_origin = None
        self.rubber_band = None
        self.original_pixmap = None
        self.crop_rect = None
        self.resize_mode = None
        self.initial_rect = None
        self._last_processed_image_path = None  # Track the last processed image path to detect new images

        # Connect signals
        self.openDirButton.clicked.connect(self.open_directory)
        self.rotateButton.clicked.connect(self.rotate_image)
        self.flipButton.clicked.connect(self.flip_image)
        self.cropButton.clicked.connect(self.toggle_crop_mode)
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
            'gamma': (0, 150,50),
            'log': (100, 200, 150),
            'glow': (1, 200, 50),
        }

        for name, (min_val, max_val, default) in slider_configs.items():
            slider = getattr(self, f"{name}Slider")
            value_label = getattr(self, f"{name}Value")
            
            slider.setMinimum(min_val)
            slider.setMaximum(max_val)
            slider.setValue(default)
            # Create a non-lambda function to avoid closure issues
            def create_value_updater(label):
                return lambda val: label.setText(f"{val/100:.3f}")

            value_updater = create_value_updater(value_label)
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
            'log_adj': self.logSlider.value() / 100,
            'clip_percentage': self.glowSlider.value() / 1000,
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
                self.image_processor = NegativeImageProcessor()
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
        if not self.current_image_path:  # if this is the first image loaded
            # Get the adjustment factors from open_image
            tint_adj_factor, white_balance_adj_factor, blacks_adj, shadows_adj, highlights_adj, whites_adj = self.image_processor.open_image(
                image_data
            )

            # Update the sliders with the calculated values
            self.tintSlider.setValue(int(tint_adj_factor * 100))
            self.whiteBalanceSlider.setValue(int(white_balance_adj_factor * 100))
            self.blacksSlider.setValue(int(blacks_adj))
            self.shadowsSlider.setValue(int(shadows_adj))
            self.highlightsSlider.setValue(int(highlights_adj))
            self.whitesSlider.setValue(int(whites_adj))
            self.glowSlider.setValue(50)

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
                
                # Get the adjustment factors from open_image
                tint_adj_factor, white_balance_adj_factor, blacks_adj, shadows_adj, highlights_adj, whites_adj = self.image_processor.open_image(self.loaded_images[file_path])
                
                # Update the sliders with the calculated values
                self.tintSlider.setValue(int(tint_adj_factor * 100))
                self.whiteBalanceSlider.setValue(int(white_balance_adj_factor * 100))
                self.blacksSlider.setValue(int(blacks_adj))
                self.shadowsSlider.setValue(int(shadows_adj))
                self.highlightsSlider.setValue(int(highlights_adj))
                self.whitesSlider.setValue(int(whites_adj))
                
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
                
                # Get the adjustment factors from open_image
                tint_adj_factor, white_balance_adj_factor, blacks_adj, shadows_adj, highlights_adj, whites_adj = self.image_processor.open_image(self.loaded_images[file_path])
                
                # Update the sliders with the calculated values
                self.tintSlider.setValue(int(tint_adj_factor * 100))
                self.whiteBalanceSlider.setValue(int(white_balance_adj_factor * 100))
                self.blacksSlider.setValue(int(blacks_adj))
                self.shadowsSlider.setValue(int(shadows_adj))
                self.highlightsSlider.setValue(int(highlights_adj))
                self.whitesSlider.setValue(int(whites_adj))
                
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

        # Store the original pixmap for cropping
        self.original_pixmap = rotated_pixmap

        # Apply crop if we have a crop rect and not in cropping mode
        if self.crop_rect is not None and not self.is_cropping:
            # We need to scale the crop rectangle from display coordinates to original image coordinates
            # The crop_rect is in display coordinates (1047 x 699) but the original image is much larger (5496 x 3670)
            
            # Calculate the scaling factor between display and original image
            display_width = self.image_label.pixmap().width()
            display_height = self.image_label.pixmap().height()
            original_width = rotated_pixmap.width()
            original_height = rotated_pixmap.height()
            
            scale_x = original_width / display_width
            scale_y = original_height / display_height
            
            # Scale the crop rectangle to original image coordinates
            scaled_rect = QRect(
                int(self.crop_rect.x() * scale_x),
                int(self.crop_rect.y() * scale_y),
                int(self.crop_rect.width() * scale_x),
                int(self.crop_rect.height() * scale_y)
            )
            
            # Ensure the scaled rect is within the image bounds
            valid_rect = QRect(
                max(0, scaled_rect.x()),
                max(0, scaled_rect.y()),
                min(scaled_rect.width(), original_width - scaled_rect.x()),
                min(scaled_rect.height(), original_height - scaled_rect.y())
            )
            
            # Apply the crop - this is the key operation that crops the image
            rotated_pixmap = rotated_pixmap.copy(valid_rect)

        label_size = self.image_label.size()
        if label_size.width() <= 0 or label_size.height() <= 0:
            return

        scaled_pixmap = rotated_pixmap.scaled(
            label_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.image_label.setPixmap(scaled_pixmap)
        
    def toggle_crop_mode(self):
        if not self.is_cropping:
            # Enter crop mode
            self.is_cropping = True
            self.cropButton.setText("Apply Crop")
            self.status_bar.showMessage("Adjust the crop rectangle and click 'Apply Crop' when done")
            
            # Store the original mouse event handlers
            self._original_mouse_press = self.image_label.mousePressEvent
            self._original_mouse_move = self.image_label.mouseMoveEvent
            self._original_mouse_release = self.image_label.mouseReleaseEvent
            
            # Enable mouse tracking for the image label
            self.image_label.setMouseTracking(True)
            self.image_label.mousePressEvent = self.crop_mouse_press
            self.image_label.mouseMoveEvent = self.crop_mouse_move
            self.image_label.mouseReleaseEvent = self.crop_mouse_release
            
            # Always update the original image for reference when entering crop mode
            if self.processed_pixmap is not None:
                transform = QTransform()
                transform.rotate(self.rotation_angle)
                if self.flip_horizontal:
                    transform.scale(-1, 1)
                self.original_pixmap = self.processed_pixmap.transformed(transform, Qt.TransformationMode.SmoothTransformation)
            
            # Create rubber band for selection if it doesn't exist
            if not self.rubber_band:
                self.rubber_band = QRubberBand(QRubberBand.Rectangle, self.image_label)
            
            # Initialize crop rectangle if not already set
            if not self.crop_rect or not self.rubber_band.isVisible():
                # Get image and label dimensions
                label_rect = self.image_label.rect()
                pixmap_size = self.image_label.pixmap().size()
                
                # Calculate scaling factors
                scale_x = label_rect.width() / pixmap_size.width()
                scale_y = label_rect.height() / pixmap_size.height()
                
                # Use the smaller scale to maintain aspect ratio
                scale = min(scale_x, scale_y)
                
                # Calculate centered image position
                img_width = pixmap_size.width() * scale
                img_height = pixmap_size.height() * scale
                img_x = (label_rect.width() - img_width) / 2
                img_y = (label_rect.height() - img_height) / 2
                
                # Create initial crop rectangle (80% of image size)
                crop_width = img_width * 0.8
                crop_height = img_height * 0.8
                crop_x = img_x + (img_width - crop_width) / 2
                crop_y = img_y + (img_height - crop_height) / 2
                
                # Set rubber band geometry
                self.rubber_band.setGeometry(QRect(
                    int(crop_x), int(crop_y),
                    int(crop_width), int(crop_height)
                ))
                
                # Calculate and store the actual crop rectangle in image coordinates
                self.update_crop_rect_from_rubber_band()
            else:
                # Restore the rubber band from existing crop_rect
                self.update_rubber_band_from_crop_rect()
            
            # Show the rubber band
            self.rubber_band.show()
        else:
            # Exit crop mode and apply crop
            self.is_cropping = False
            self.cropButton.setText("Crop")
            self.status_bar.showMessage("Crop applied")
            
            # Restore original mouse event handlers
            self.image_label.setMouseTracking(False)
            if hasattr(self, '_original_mouse_press'):
                self.image_label.mousePressEvent = self._original_mouse_press
            else:
                self.image_label.mousePressEvent = None
                
            if hasattr(self, '_original_mouse_move'):
                self.image_label.mouseMoveEvent = self._original_mouse_move
            else:
                self.image_label.mouseMoveEvent = None
                
            if hasattr(self, '_original_mouse_release'):
                self.image_label.mouseReleaseEvent = self._original_mouse_release
            else:
                self.image_label.mouseReleaseEvent = None
            
            # Hide rubber band
            if self.rubber_band:
                self.rubber_band.hide()
            
            # Make sure we have a valid crop rectangle
            if self.crop_rect and self.crop_rect.isValid() and self.crop_rect.width() > 0 and self.crop_rect.height() > 0:
                # Apply crop
                pass
            else:
                # Invalid crop rectangle
                pass
                
            # Apply the crop
            self.transform_and_display()
    
    def update_crop_rect_from_rubber_band(self):
        """Convert rubber band coordinates to image coordinates and update crop_rect"""
        if not self.rubber_band or not self.original_pixmap:
            return
            
        rubber_band_rect = self.rubber_band.geometry()
        label_rect = self.image_label.rect()
        pixmap_size = self.image_label.pixmap().size()
        
        # Get dimensions for calculations
        
        # Calculate scaling factors
        scale_x = label_rect.width() / pixmap_size.width()
        scale_y = label_rect.height() / pixmap_size.height()
        
        # Use the smaller scale to maintain aspect ratio
        scale = min(scale_x, scale_y)
        
        # Calculate centered image position
        img_width = pixmap_size.width() * scale
        img_height = pixmap_size.height() * scale
        img_x = (label_rect.width() - img_width) / 2
        img_y = (label_rect.height() - img_height) / 2
        
        # Create image rect
        image_rect = QRect(int(img_x), int(img_y), int(img_width), int(img_height))
        
        # Calculate the relative position of the rubber band within the displayed image
        rel_x = (rubber_band_rect.x() - image_rect.x()) / scale
        rel_y = (rubber_band_rect.y() - image_rect.y()) / scale
        rel_width = rubber_band_rect.width() / scale
        rel_height = rubber_band_rect.height() / scale
        
        
        # Ensure coordinates are within bounds
        rel_x = max(0, min(rel_x, pixmap_size.width()))
        rel_y = max(0, min(rel_y, pixmap_size.height()))
        rel_width = min(rel_width, pixmap_size.width() - rel_x)
        rel_height = min(rel_height, pixmap_size.height() - rel_y)
        
        # Store the crop rectangle in image coordinates
        self.crop_rect = QRect(int(rel_x), int(rel_y), int(rel_width), int(rel_height))
        
        # Show crop dimensions in status bar
        self.status_bar.showMessage(f"Crop selection: {int(rel_width)}x{int(rel_height)} pixels")
    
    def update_rubber_band_from_crop_rect(self):
        """Convert image coordinates to rubber band coordinates and update rubber band"""
        if not self.rubber_band or not self.crop_rect or not self.original_pixmap:
            return
            
        label_rect = self.image_label.rect()
        pixmap_size = self.image_label.pixmap().size()
        
        # Calculate scaling factors
        scale_x = label_rect.width() / pixmap_size.width()
        scale_y = label_rect.height() / pixmap_size.height()
        
        # Use the smaller scale to maintain aspect ratio
        scale = min(scale_x, scale_y)
        
        # Calculate centered image position
        img_width = pixmap_size.width() * scale
        img_height = pixmap_size.height() * scale
        img_x = (label_rect.width() - img_width) / 2
        img_y = (label_rect.height() - img_height) / 2
        
        # Create image rect
        image_rect = QRect(int(img_x), int(img_y), int(img_width), int(img_height))
        
        # Convert crop rect to rubber band coordinates
        rb_x = image_rect.x() + (self.crop_rect.x() * scale)
        rb_y = image_rect.y() + (self.crop_rect.y() * scale)
        rb_width = self.crop_rect.width() * scale
        rb_height = self.crop_rect.height() * scale
        
        # Set rubber band geometry
        self.rubber_band.setGeometry(QRect(
            int(rb_x), int(rb_y),
            int(rb_width), int(rb_height)
        ))
    
    def crop_mouse_press(self, event):
        if not self.is_cropping or not self.original_pixmap or not self.rubber_band:
            return
        
        # Get the rubber band geometry
        rect = self.rubber_band.geometry()
        
        # Determine which part of the crop rectangle was clicked
        # (edges, corners, or inside for moving)
        margin = 10  # pixels for edge/corner detection
        
        # Check corners first (they take precedence over edges)
        if abs(event.pos().x() - rect.left()) <= margin and abs(event.pos().y() - rect.top()) <= margin:
            self.resize_mode = "top-left"
        elif abs(event.pos().x() - rect.right()) <= margin and abs(event.pos().y() - rect.top()) <= margin:
            self.resize_mode = "top-right"
        elif abs(event.pos().x() - rect.left()) <= margin and abs(event.pos().y() - rect.bottom()) <= margin:
            self.resize_mode = "bottom-left"
        elif abs(event.pos().x() - rect.right()) <= margin and abs(event.pos().y() - rect.bottom()) <= margin:
            self.resize_mode = "bottom-right"
        # Then check edges
        elif abs(event.pos().x() - rect.left()) <= margin:
            self.resize_mode = "left"
        elif abs(event.pos().x() - rect.right()) <= margin:
            self.resize_mode = "right"
        elif abs(event.pos().y() - rect.top()) <= margin:
            self.resize_mode = "top"
        elif abs(event.pos().y() - rect.bottom()) <= margin:
            self.resize_mode = "bottom"
        # If inside the rectangle, it's a move operation
        elif rect.contains(event.pos()):
            self.resize_mode = "move"
        else:
            self.resize_mode = None
            return
        
        # Store the initial position and rectangle
        self.crop_origin = event.pos()
        self.initial_rect = rect
    
    def crop_mouse_move(self, event):
        if not self.is_cropping or not self.rubber_band or not self.crop_origin or not self.resize_mode:
            return
        
        # Get the current mouse position
        current_pos = event.pos()
        
        # Calculate the delta from the original position
        dx = current_pos.x() - self.crop_origin.x()
        dy = current_pos.y() - self.crop_origin.y()
        
        # Get the initial rectangle
        rect = QRect(self.initial_rect)
        
        # Maintain aspect ratio
        aspect_ratio = rect.width() / rect.height()
        
        # Update the rectangle based on the resize mode
        if self.resize_mode == "move":
            # Move the entire rectangle
            rect.translate(dx, dy)
        elif self.resize_mode == "top-left":
            # Resize from top-left corner
            new_width = rect.width() - dx
            new_height = new_width / aspect_ratio
            rect.setLeft(rect.right() - new_width)
            rect.setTop(rect.bottom() - new_height)
        elif self.resize_mode == "top-right":
            # Resize from top-right corner
            new_width = rect.width() + dx
            new_height = new_width / aspect_ratio
            rect.setRight(rect.left() + new_width)
            rect.setTop(rect.bottom() - new_height)
        elif self.resize_mode == "bottom-left":
            # Resize from bottom-left corner
            new_width = rect.width() - dx
            new_height = new_width / aspect_ratio
            rect.setLeft(rect.right() - new_width)
            rect.setBottom(rect.top() + new_height)
        elif self.resize_mode == "bottom-right":
            # Resize from bottom-right corner
            new_width = rect.width() + dx
            new_height = new_width / aspect_ratio
            rect.setRight(rect.left() + new_width)
            rect.setBottom(rect.top() + new_height)
        elif self.resize_mode == "left":
            # Resize from left edge
            new_width = rect.width() - dx
            new_height = new_width / aspect_ratio
            rect.setLeft(rect.right() - new_width)
            # Center vertically
            center_y = rect.center().y()
            rect.setHeight(new_height)
            rect.moveCenter(QPoint(rect.center().x(), center_y))
        elif self.resize_mode == "right":
            # Resize from right edge
            new_width = rect.width() + dx
            new_height = new_width / aspect_ratio
            rect.setRight(rect.left() + new_width)
            # Center vertically
            center_y = rect.center().y()
            rect.setHeight(new_height)
            rect.moveCenter(QPoint(rect.center().x(), center_y))
        elif self.resize_mode == "top":
            # Resize from top edge
            new_height = rect.height() - dy
            new_width = new_height * aspect_ratio
            rect.setTop(rect.bottom() - new_height)
            # Center horizontally
            center_x = rect.center().x()
            rect.setWidth(new_width)
            rect.moveCenter(QPoint(center_x, rect.center().y()))
        elif self.resize_mode == "bottom":
            # Resize from bottom edge
            new_height = rect.height() + dy
            new_width = new_height * aspect_ratio
            rect.setBottom(rect.top() + new_height)
            # Center horizontally
            center_x = rect.center().x()
            rect.setWidth(new_width)
            rect.moveCenter(QPoint(center_x, rect.center().y()))
        
        # Ensure the rectangle stays within the image bounds
        label_rect = self.image_label.rect()
        pixmap_size = self.image_label.pixmap().size()
        
        # Calculate scaling factors
        scale_x = label_rect.width() / pixmap_size.width()
        scale_y = label_rect.height() / pixmap_size.height()
        
        # Use the smaller scale to maintain aspect ratio
        scale = min(scale_x, scale_y)
        
        # Calculate centered image position
        img_width = pixmap_size.width() * scale
        img_height = pixmap_size.height() * scale
        img_x = (label_rect.width() - img_width) / 2
        img_y = (label_rect.height() - img_height) / 2
        
        # Create image rect
        image_rect = QRect(int(img_x), int(img_y), int(img_width), int(img_height))
        
        # Constrain the crop rectangle to the image bounds
        if rect.left() < image_rect.left():
            rect.setLeft(image_rect.left())
        if rect.top() < image_rect.top():
            rect.setTop(image_rect.top())
        if rect.right() > image_rect.right():
            rect.setRight(image_rect.right())
        if rect.bottom() > image_rect.bottom():
            rect.setBottom(image_rect.bottom())
        
        # Update the rubber band
        self.rubber_band.setGeometry(rect)
        
        # Update the crop rectangle
        self.update_crop_rect_from_rubber_band()
    
    def crop_mouse_release(self, event):
        # Reset the resize mode
        self.resize_mode = None

    def display_image(self):
        if not self.file_manager or not self.image_processor or self.image_processor.current_rgb is None:
            self.image_label.clear()
            self.processed_pixmap = None  # Reset pixmap
            self.original_pixmap = None   # Reset original pixmap
            self.crop_rect = None         # Reset crop rectangle
            self.resize_mode = None       # Reset resize mode
            self.initial_rect = None      # Reset initial rect
            return

        try:
            # Get current adjustments
            adjustments = self.get_slider_values()

            # Process the image with current adjustments
            processed_array = self.image_processor.process_image(
                **adjustments
            )
            
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
                self.original_pixmap = None   # Reset original pixmap
                self.crop_rect = None         # Reset crop rectangle
                return

            # Create and store the processed pixmap
            self.processed_pixmap = QPixmap.fromImage(img_qt)
            
            # Only reset crop state when loading a completely new image, not when just adjusting sliders
            # We can detect slider changes vs. new image by checking if we're in the process_image_update method
            # which is called from trigger_update when sliders change
            if hasattr(self, '_last_processed_image_path') and self._last_processed_image_path != self.current_image_path:
                # New image detected, resetting crop
                if self.is_cropping:
                    self.toggle_crop_mode()  # Exit crop mode
                self.crop_rect = None
            
            # Store the current image path for comparison next time
            self._last_processed_image_path = self.current_image_path
            
            # Apply transformations and display
            self.transform_and_display()
            
            # Update status
            self.status_bar.showMessage(f"Displaying image {self.image_index + 1} of {len(self.raw_file_paths)}")

        except Exception as e:
            self.status_bar.showMessage(f"Error displaying image: {str(e)}")
            print(f"Display Error: {e}")
            self.image_label.clear()
            self.processed_pixmap = None  # Reset pixmap
            self.original_pixmap = None   # Reset original pixmap
            self.crop_rect = None         # Reset crop rectangle
            self.resize_mode = None       # Reset resize mode
            self.initial_rect = None      # Reset initial rect

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
            
            # If we have a crop, we need to apply it to the image data before saving
            if self.crop_rect is not None and self.processed_pixmap is not None:
                # Create a QImage from the processed array
                processed_array = self.image_processor.current_rgb
                processed_array = np.clip(processed_array / 256, 0, 255).astype(np.uint8)
                
                height, width, channel = processed_array.shape
                bytes_per_line = 3 * width
                img_qt = QImage(processed_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
                
                # Create a pixmap and apply transformations
                pixmap = QPixmap.fromImage(img_qt)
                transform = QTransform()
                transform.rotate(self.rotation_angle)
                if self.flip_horizontal:
                    transform.scale(-1, 1)
                
                transformed_pixmap = pixmap.transformed(transform, Qt.TransformationMode.SmoothTransformation)
                
                # We need to scale the crop rectangle from display coordinates to original image coordinates
                # Calculate the scaling factor between display and original image
                display_width = self.image_label.pixmap().width()
                display_height = self.image_label.pixmap().height()
                original_width = transformed_pixmap.width()
                original_height = transformed_pixmap.height()
                
                scale_x = original_width / display_width
                scale_y = original_height / display_height
                
                
                # Scale the crop rectangle to original image coordinates
                scaled_rect = QRect(
                    int(self.crop_rect.x() * scale_x),
                    int(self.crop_rect.y() * scale_y),
                    int(self.crop_rect.width() * scale_x),
                    int(self.crop_rect.height() * scale_y)
                )
                
                
                # Ensure the scaled rect is within the image bounds
                valid_rect = QRect(
                    max(0, scaled_rect.x()),
                    max(0, scaled_rect.y()),
                    min(scaled_rect.width(), original_width - scaled_rect.x()),
                    min(scaled_rect.height(), original_height - scaled_rect.y())
                )
                
                
                # Apply the crop
                cropped_pixmap = transformed_pixmap.copy(valid_rect)
                
                # Instead of using the displayed pixmap, we'll apply the crop to the original processed image
                # This ensures consistent color processing between cropped and non-cropped images
                
                # Get the original processed image data
                processed_array = self.image_processor.current_rgb
                
                # Create a PIL image from the processed array for easier manipulation
                processed_image = np.clip(processed_array / 256, 0, 255).astype(np.uint8)
                pil_image = Image.fromarray(processed_image)
                
                # Apply rotation and flip if needed
                if self.rotation_angle != 0:
                    pil_image = pil_image.rotate(-self.rotation_angle, expand=True)
                if self.flip_horizontal:
                    pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
                
                # Calculate the crop coordinates in the original image space
                img_width, img_height = pil_image.size
                
                # Scale the crop rectangle to match the original image dimensions
                display_width = self.image_label.pixmap().width()
                display_height = self.image_label.pixmap().height()
                
                scale_x = img_width / display_width
                scale_y = img_height / display_height
                
                # Scale the crop rectangle to original image coordinates
                scaled_rect = QRect(
                    int(self.crop_rect.x() * scale_x),
                    int(self.crop_rect.y() * scale_y),
                    int(self.crop_rect.width() * scale_x),
                    int(self.crop_rect.height() * scale_y)
                )
                
                # Ensure the scaled rect is within the image bounds
                valid_rect = QRect(
                    max(0, scaled_rect.x()),
                    max(0, scaled_rect.y()),
                    min(scaled_rect.width(), img_width - scaled_rect.x()),
                    min(scaled_rect.height(), img_height - scaled_rect.y())
                )
                
                # Crop the PIL image
                cropped_pil = pil_image.crop((
                    valid_rect.x(),
                    valid_rect.y(),
                    valid_rect.x() + valid_rect.width(),
                    valid_rect.y() + valid_rect.height()
                ))
                
                # Convert PIL image to numpy array
                cropped_array = np.array(cropped_pil)
                
                # Use the thread with is_preprocessed=True for cropped images
                self.image_saver_thread = ImageSaverThread(cropped_array, save_path, 0, False, is_preprocessed=True)
            else:
                # Save the original image with rotation/flip
                self.image_saver_thread = ImageSaverThread(self.image_processor.current_rgb, save_path, self.rotation_angle, self.flip_horizontal)
            
            # Connect signals and start the thread
            self.image_saver_thread.finished_saving.connect(self.saving_finished)
            self.image_saver_thread.error_saving.connect(self.saving_error)
            self.image_saver_thread.start()
            self.status_bar.showMessage(f"Saving image: {file_name}...")

        except Exception as e:
            self.status_bar.showMessage(f"Error preparing to save: {str(e)}")
            print(f"Save Error: {e}")

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
            'gamma': (0, 150, 50),
            'log': (100, 200, 150),
            'glow': (1, 200, 50),
        }

        for name, (min_val, max_val, default) in slider_configs.items():
            slider = getattr(self, f"{name}Slider")
            slider.setValue(default)

        # Reset crop state
        if self.is_cropping:
            self.toggle_crop_mode()  # Exit crop mode if active
        self.crop_rect = None
        
        # Hide rubber band if visible
        if self.rubber_band and self.rubber_band.isVisible():
            self.rubber_band.hide()

        # Refresh the image to reflect the reset sliders and crop
        self.display_image()
        self.status_bar.showMessage("Adjustments and crop have been reset")


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
