import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QSlider, QFileDialog, 
                            QScrollArea, QGroupBox, QStatusBar, QFrame)
from PyQt6.QtCore import Qt, QSize, QTimer
from PyQt6.QtGui import QPixmap, QImage, QIcon, QFont, QFontDatabase, QTransform
import numpy as np
from PIL import Image
from neg_processor import NumbaOptimizedNegativeImageProcessor

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
                background-color: {GRAY_BG};
                color: {TEXT_COLOR};
                padding: 6px;
                font-size: 14px; /* Slightly larger status bar font */
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

        # Create sidebar
        # Sidebar with explicit text colors
        sidebar = QWidget()
        sidebar.setFixedWidth(320)
        
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
        sidebar.setFixedWidth(380)
        sidebar_layout.setContentsMargins(30, 30, 30, 30)
        sidebar_layout.setSpacing(24)

        # Add a subtle background color to the sidebar
        sidebar.setStyleSheet(f"background-color: {GRAY_BG};")  # Use your GRAY_BG color

        # Dashboard title
        # Dashboard title and logo
        title_label = QLabel("Orpheus Negative Lab")
        title_label.setStyleSheet(f"""
            QLabel {{
                color: {TEXT_COLOR};
                font-size: 28px;
                font-weight: 600;
                padding: 18px 0;
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
        adjustments_layout = QVBoxLayout()  # Or QHBoxLayout if you want horizontal layout
        adjustments_group.setLayout(adjustments_layout)
        # Create sliders
        self.sliders = {}
        slider_configs = {
            'R': (-0.5, 0.5, 0),
            'G': (-0.5, 0.5, 0),
            'B': (-0.5, 0.5, -0.3),
            'Exposure': (-1, 1, 0),
            'Gamma': (0.1, 4, 2.2)
        }

        for name, (min_val, max_val, default) in slider_configs.items():
            slider_widget = QWidget()
            slider_layout = QVBoxLayout(slider_widget)
            slider_layout.setSpacing(4)
            
            header_layout = QHBoxLayout()
            label = QLabel(name)
            value_label = QLabel(f"{default:.2f}")
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            value_label.setStyleSheet(f"color: {'#555555'};")
            
            header_layout.addWidget(label)
            header_layout.addWidget(value_label)
            
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(int(min_val * 100))
            slider.setMaximum(int(max_val * 100))
            slider.setValue(int(default * 100))
            
            slider.valueChanged.connect(lambda value, label=name: self.update_slider_label(label))
            slider.sliderReleased.connect(self.trigger_update)
            
            slider_layout.addLayout(header_layout)
            slider_layout.addWidget(slider)
            
            self.sliders[name] = (slider, value_label)
            adjustments_layout.addWidget(slider_widget)

        sidebar_layout.addWidget(adjustments_group)

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
                padding: 8px 18px;
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
                    padding: 6px 10px;
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

        # Set window size
        self.resize(1400, 900)

    def get_slider_values(self):
        return {
            'r_adjust': self.sliders['R'][0].value() / 100,
            'g_adjust': self.sliders['G'][0].value() / 100,
            'b_adjust': self.sliders['B'][0].value() / 100,
            'exposure': self.sliders['Exposure'][0].value() / 100,
            'gamma': self.sliders['Gamma'][0].value() / 100
        }

    def update_slider_label(self, name):
        slider, label = self.sliders[name]
        label.setText(f"{slider.value() / 100:.2f}")

    def trigger_update(self):
        self.update_timer.start(100)

    def process_image_update(self):
        self.display_image()

    def open_directory(self):
        self.directory_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if self.directory_path:
            try:
                self.processor = NumbaOptimizedNegativeImageProcessor(self.directory_path)
                self.image_index = 0

                # Find the first valid image in the directory
                while self.image_index < len(self.processor.file_paths):
                    file_path = self.processor.get_file_path(self.image_index)
                    try:
                        self.processor.open_image(file_path)
                        self.display_image()
                        self.status_bar.showMessage(f"Loaded directory: {self.directory_path}")
                        break  # Found a valid image, exit the loop
                    except Exception as e:
                        print(f"Skipping invalid file: {file_path} - {e}")  # Print to console for debugging
                        self.image_index += 1  # Move to the next file

                if self.image_index == len(self.processor.file_paths):  # No valid images found
                    self.status_bar.showMessage("No valid image files found in the directory.")
                    self.image_label.clear()  # Clear the image area
                    self.processor = None # Reset the processor to avoid errors if the user tries to interact with it
                    return # exit the function

            except Exception as e:
                self.status_bar.showMessage(f"Error loading directory: {str(e)}")
                self.image_label.clear()  # Clear the image area
                self.processor = None # Reset the processor to avoid errors if the user tries to interact with it
                return # exit the function

    def prev_image(self):
        if self.processor and self.image_index > 0:
            self.save_current_image()
            self.image_index -= 1

            while self.image_index >= 0: # loop backwards through files to find a valid one
                file_path = self.processor.get_file_path(self.image_index)
                try:
                    self.processor.open_image(file_path)
                    self.display_image()
                    self.status_bar.showMessage(f"Showing image {self.image_index + 1} of {len(self.processor.file_paths)}")
                    return # exit the function once a valid image is found
                except Exception as e:
                    print(f"Skipping invalid file: {file_path} - {e}")
                    self.image_index -= 1
            
            # if we get here, no valid images were found
            self.status_bar.showMessage("No valid previous image found.")
            self.image_index = 0 # reset index
            file_path = self.processor.get_file_path(self.image_index) # try opening the first file again
            try:
                self.processor.open_image(file_path)
                self.display_image()
                self.status_bar.showMessage(f"Showing image {self.image_index + 1} of {len(self.processor.file_paths)}")
            except:
                self.image_label.clear()

    def next_image(self):
        if self.processor and self.image_index < len(self.processor.file_paths) - 1:
            self.save_current_image()
            self.image_index += 1

            while self.image_index < len(self.processor.file_paths): # loop forward through files to find a valid one
                file_path = self.processor.get_file_path(self.image_index)
                try:
                    self.processor.open_image(file_path)
                    self.display_image()
                    self.status_bar.showMessage(f"Showing image {self.image_index + 1} of {len(self.processor.file_paths)}")
                    return # exit the function once a valid image is found
                except Exception as e:
                    print(f"Skipping invalid file: {file_path} - {e}")
                    self.image_index += 1
            
            # if we get here, no valid images were found
            self.status_bar.showMessage("No valid next image found.")
            self.image_index = len(self.processor.file_paths) - 1 # reset index
            file_path = self.processor.get_file_path(self.image_index) # try opening the last file again
            try:
                self.processor.open_image(file_path)
                self.display_image()
                self.status_bar.showMessage(f"Showing image {self.image_index + 1} of {len(self.processor.file_paths)}")
            except:
                self.image_label.clear()

    def rotate_image(self):
        self.rotation_angle = (self.rotation_angle - 90) % 360
        self.display_image()

    def flip_image(self):
        self.flip_horizontal = not self.flip_horizontal  # Toggle flip state
        self.display_image()

    def display_image(self):
        if not self.processor or self.processor.current_rgb is None:
            self.image_label.clear()
            return

        try:
            values = self.get_slider_values()
            processed_image = self.processor.process_image(
                values['r_adjust'], values['g_adjust'], values['b_adjust'],
                values['gamma'], values['exposure']
            )

            image = Image.fromarray((processed_image / 256).astype(np.uint8))
            img_qt = QImage(image.tobytes(), image.width, image.height, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(img_qt)

            transform = QTransform()
            transform.rotate(self.rotation_angle)

            if self.flip_horizontal:  # Apply flip if needed
                transform.scale(-1, 1)  # Horizontal flip

            rotated_pixmap = pixmap.transformed(transform)

            scaled_pixmap = rotated_pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            self.image_label.setPixmap(scaled_pixmap)

        except Exception as e:
            self.status_bar.showMessage(f"Error displaying image: {str(e)}")
            self.image_label.clear()

    def reset_sliders(self):
        # Reset all sliders to their default values
        slider_configs = {  # Get your slider configurations
            'R': (-0.5, 0.5, 0),
            'G': (-0.5, 0.5, 0),
            'B': (-0.5, 0.5, -0.3),
            'Exposure': (-1, 1, 0),
            'Gamma': (0.1, 4, 2.2)
        }

        for name, (min_val, max_val, default) in slider_configs.items():
            slider, label = self.sliders[name]
            slider.setValue(int(default * 100))  # Set slider to default value
            label.setText(f"{default:.2f}")  # Update the label

        self.display_image() # Refresh the image to reflect the reset sliders

    def save_current_image(self):
        if not self.processor or self.processor.current_rgb is None:
            return

        positives_dir = os.path.join(self.directory_path, "positives")
        os.makedirs(positives_dir, exist_ok=True)

        file_name = os.path.basename(self.processor.get_file_path(self.image_index)).lower()
        for ext in [".dng", ".cr2", ".cr3", ".raw", ".nef"]:
            file_name = file_name.replace(ext, ".jpg")
        save_path = os.path.join(positives_dir, file_name)

        processed_image = self.processor.current_rgb
        image = Image.fromarray((processed_image / 256).astype(np.uint8))
        image.save(save_path, "JPEG", quality=95)
        self.status_bar.showMessage(f"Saved image: {file_name}")

def main():
    app = QApplication(sys.argv)
    font = QFont('Roboto') # Use Noto Sans
    app.setFont(font)

    if app.font().family() != 'Roboto': # Check if Noto Sans was loaded
        print("Roboto not found, using a fallback font.")
        font = QFont('Arial')  # Fallback to Arial or another common font
        app.setFont(font)
    else:
        app.setFont(font)
    
    # Set the base style for the entire application
    app.setStyleSheet(f"""
        * {{
            color: #333333;
        }}
    """)
    
    window = ModernNegativeImageGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()