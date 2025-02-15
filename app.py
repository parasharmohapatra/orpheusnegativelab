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
        self.setWindowTitle("Orpheus Negative Lab")
        
        # Minimalistic Color Palette
        BLUE = "#2196F3"  # A bit less saturated blue
        BLUE_HOVER = "#1E88E5"
        GRAY_BG = "#F5F5F5"  # Very light gray
        BORDER_COLOR = "#EEEEEE"  # Even lighter border
        TEXT_COLOR = "#333"  # Dark gray text
        WHITE = "#FFFFFF"

        # Base styles
        app = QApplication.instance()
        app.setStyleSheet(f"""
            * {{
                color: {TEXT_COLOR};
                font-family: 'Roboto', sans-serif;
            }}
        """)

        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {WHITE};
            }}
            QLabel {{
                color: {TEXT_COLOR};
                font-size: 16px; /* Increased base font size */
            }}
            QPushButton {{
                background-color: {BLUE};
                color: #555;
                border: none; /* Removed border for minimalist look */
                padding: 10px 20px;
                border-radius: 8px;
                font-size: 16px;
                min-width: 120px;
                margin: 6px;
            }}
            QPushButton:hover {{
                background-color: {BLUE_HOVER};
            }}
            QPushButton:pressed {{
                background-color: {BLUE_HOVER};
                padding: 9px 19px; /* Slightly reduced padding */
            }}
            QSlider::groove:horizontal {{
                border: {GRAY_BG}; /* Removed border */
                height: 8px;
                background: {WHITE}; /* Light gray groove */
                margin: 0px;
                border-radius: 4px;
            }}
            QSlider::handle:horizontal {{
                background: {'#555555'};
                border: none;
                width: 20px;
                margin: -6px 0;
                border-radius: 10px;
            }}
            QGroupBox {{
                background-color: {WHITE};
                border: none; /* Removed border */
                margin-top: 20px;
                font-weight: 500;
                padding: 24px;
                color: {TEXT_COLOR};
            }}
            QGroupBox::title {{
                color: {TEXT_COLOR};
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 8px;
                font-size: 18px; /* Increased groupbox title font */
            }}
            QStatusBar {{
                background-color: #fdf0e7;  /* Match sidebar color */
                color: {TEXT_COLOR};
                padding: 6px;
                font-size: 14px;
            }}
            QScrollArea {{
                border: none; /* Removed border */
                background-color: {WHITE};
            }}
        """)

        # Initialize debounce timer
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.process_image_update)

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Initialize variables
        self.processor = None
        self.image_index = 0
        self.current_image = None
        self.directory_path = ""
        self.loaded_images = {}
        self.current_image_path = None
        self.raw_file_paths = [] # List to store only raw file paths
        self.image_saver_thread = None # Add the saver thread attribute
        self.image_loader_thread = None  # Add thread attribute
        self.processed_pixmap = None  # Store the processed pixmap
        self.progress_bar = QProgressBar()  # Create progress bar
        self.progress_bar.setFixedHeight(8)  # Make it even slimmer
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 4px;
                background-color: #f0f0f0;
                text-align: center;
                margin: 0px 15px;  /* Add some margin on the sides */
                max-width: 400px;  /* Limit the width */
            }
            QProgressBar::chunk {
                background-color: #2196F3;  /* Match the blue theme */
                border-radius: 4px;
            }
        """)

        # Create sidebar
        # Sidebar with explicit text colors
        sidebar = QWidget()
        
        sidebar.setStyleSheet(f"""
            QWidget {{
                background-color: {GRAY_BG};
                color: {TEXT_COLOR};
            }}
            QLabel {{
                color: {TEXT_COLOR};
            }}
        """)
        sidebar_layout = QVBoxLayout(sidebar)
         # Sidebar adjustments (more aesthetic)
        sidebar_layout.setContentsMargins(30, 30, 30, 30)
        sidebar_layout.setSpacing(24)

        # Add a subtle background color to the sidebar
        sidebar.setStyleSheet(f"background-color: {'#fdf0e7'};")  # Use your GRAY_BG color

        # Dashboard title
        # Dashboard title and logo
        title_label = QLabel("Orpheus Negative Lab")
        title_label.setStyleSheet(f"""
            QLabel {{
                color: {TEXT_COLOR};
                font-size: 28px;
                font-weight: 600;
                padding: 14px 26px;
            }}
        """)
        sidebar_layout.addWidget(title_label)
        

        # Open Directory Button (with border)
        self.open_button = QPushButton("Open Directory")
        self.open_button.setStyleSheet("""
            QPushButton {
                border: 1px solid #D3D3D3; /* Light gray border */
                border-radius: 8px; /* Rounded corners */
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #EEEEEE; /* Subtle hover effect */
            }
        """)
        self.open_button.clicked.connect(self.open_directory)
        sidebar_layout.addWidget(self.open_button)

        # Separator (more visible)
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("background-color: #555555; height: 1px;")  # Thicker, lighter gray
        sidebar_layout.addWidget(separator)
        sidebar_layout.setSpacing(12)

        # Adjustments Group (renamed and with better styling)
        adjustments_group = QGroupBox("")  # Renamed
        adjustments_group.setStyleSheet("""
            QGroupBox {
                border: none; /* Removed border */
                margin-top: 5px;
                font-weight: 500;
                padding: 12px;
                font-size: 18px;
            }
            QGroupBox::title {
                color: #555; /* Slightly darker title */
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 10px;
                font-size: 18px;
            }
        """)
        # Create a layout for the adjustments group (important!)
        # Use QGridLayout for slider layout
        adjustments_layout = QGridLayout()  # Use GridLayout
        adjustments_group.setLayout(adjustments_layout)
        # Create sliders
        self.sliders = {}
        slider_configs = {
            'R': (-0.5, 0.5, 0),
            'G': (-0.5, 0.5, -0.1),
            'B': (-0.5, 0.5, -0.2),
            'Brightness': (-1, 1, 0),
            'Gamma': (.1, 2, 1), 
            'Highlights': (-0.5, 0.5, 0),
            'Shadows': (-0.5, 0.5, 0),
        }

        row = 0  # Row counter for grid layout
        for name, (min_val, max_val, default) in slider_configs.items():
            label = QLabel(name)
            label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

            value_label = QLabel(f"{default:.2f}")
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            value_label.setStyleSheet(f"color: {'#555555'}; font-size: 14px;")

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(int(min_val * 100))
            slider.setMaximum(int(max_val * 100))
            slider.setValue(int(default * 100))

            # Correctly connect the signal with a lambda function
            slider.valueChanged.connect(lambda value, lbl=value_label: lbl.setText(f"{value / 100:.2f}"))
            slider.sliderReleased.connect(self.trigger_update)

            adjustments_layout.addWidget(label, row, 0)
            adjustments_layout.addWidget(slider, row, 1)
            adjustments_layout.addWidget(value_label, row, 2)

            self.sliders[name] = (slider, value_label)  # Store both slider and label
            row += 1

        sidebar_layout.addWidget(adjustments_group)
        sidebar_layout.setSpacing(8)

        # Rotate Button
        self.rotate_button = QPushButton("Rotate")
        self.rotate_button.setStyleSheet("""
            QPushButton {
                border: 1px solid #D3D3D3;
                border-radius: 8px;
                padding: 8px 16px;
                color: #555;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #EEEEEE;
            }
        """)
        self.rotate_button.clicked.connect(self.rotate_image)

        # Flip Button
        self.flip_button = QPushButton("Flip")
        self.flip_button.setStyleSheet("""
            QPushButton {
                border: 1px solid #D3D3D3;
                border-radius: 8px;
                padding: 8px 16px;
                color: #555;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #EEEEEE;
            }
        """)
        self.flip_button.clicked.connect(self.flip_image)

        # Reset Button
        self.reset_button = QPushButton("Reset Adjustments")
        self.reset_button.setStyleSheet("""
            QPushButton {
                border: 1px solid #D3D3D3;
                border-radius: 8px;
                padding: 8px 22px;
                color: #555;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #EEEEEE;
            }
        """)
        self.reset_button.clicked.connect(self.reset_sliders)
        
        # Arrange buttons (Rotate and Flip horizontally, Reset below)
        rotate_flip_layout = QHBoxLayout()  # Horizontal layout for Rotate and Flip
        rotate_flip_layout.addWidget(self.rotate_button)
        rotate_flip_layout.addWidget(self.flip_button)

        rotate_flip_reset_layout = QVBoxLayout()  # Vertical layout for all three
        rotate_flip_reset_layout.addLayout(rotate_flip_layout)  # Add Rotate/Flip layout
        rotate_flip_reset_layout.addWidget(self.reset_button)  # Add Reset button below

        sidebar_layout.addLayout(rotate_flip_reset_layout)

        self.rotation_angle = 0
        self.flip_horizontal = False  # Initialize flip state
        
        # Navigation buttons
        nav_widget = QWidget()
        nav_layout = QHBoxLayout(nav_widget)
        nav_layout.setSpacing(8)

        # Navigation buttons (with borders)
        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        for button in [self.prev_button, self.next_button]:
            button.setStyleSheet("""
                QPushButton {
                    border: 1px solid #D3D3D3;
                    border-radius: 8px;
                    padding: 8px 16px;
                    color: #555;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #EEEEEE;
                }
            """)
        self.prev_button.clicked.connect(self.prev_image)
        self.next_button.clicked.connect(self.next_image)

        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        sidebar_layout.addWidget(nav_widget)

        # Add spacer
        sidebar_layout.addStretch()

        # Add sidebar to main layout
        layout.addWidget(sidebar)

        # Image display area
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_widget.setStyleSheet("background-color: white;")
        content_layout.setContentsMargins(20, 20, 20, 20)
        layout.setStretchFactor(sidebar, 0)
        layout.setStretchFactor(content_widget, 1)
        
        # Scroll area for image
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll_area.setWidget(self.image_label)
        
        content_layout.addWidget(scroll_area)
        layout.addWidget(content_widget, stretch=1)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.addPermanentWidget(self.progress_bar) # Add to status bar
        self.progress_bar.hide()  # Initially hide the progress bar

        # Set window size and properties
        self.showMaximized()  # Show maximized window with normal window controls
        #self.setFixedSize(self.size())  # Lock the window size

    def get_slider_values(self):
        return {
            'r_adjust': self.sliders['R'][0].value() / 100,
            'g_adjust': self.sliders['G'][0].value() / 100,
            'b_adjust': self.sliders['B'][0].value() / 100,
            'brightness': self.sliders['Brightness'][0].value() / 100,
            'gamma': self.sliders['Gamma'][0].value() / 100, 
            'highlights': self.sliders['Highlights'][0].value() / 100,
            'shadows': self.sliders['Shadows'][0].value() / 100,
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
                values['brightness'], values['gamma'], values['highlights'], values['shadows']
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
            'G': (-0.5, 0.5, -0.1),
            'B': (-0.5, 0.5, -0.2),
            'Brightness': (-1, 1, 0),
            'Gamma': (0.1, 2, 1), 
            'Highlights': (-0.5, 0.5, 0),
            'Shadows': (-0.5, 0.5, 0),
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
