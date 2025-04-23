import os
import sys
import cv2
import rawpy
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QLabel, QPushButton, QSlider, QFileDialog, QSplitter, QFrame,
                            QSpacerItem, QSizePolicy, QGridLayout, QScrollArea, QMessageBox,
                            QAction, QToolBar, QStatusBar, QMenu, QCheckBox)
from PyQt5.QtGui import QPixmap, QImage, QWheelEvent, QTransform, QCursor, QKeyEvent, QIcon, QFontDatabase
from PyQt5.QtCore import Qt, QBuffer, QIODevice, QTimer, QPoint, QRectF

class ToneCurveProcessor:
    def __init__(self, input_path=None, output_path=None):
        """
        Initialize the ToneCurveProcessor.

        Parameters:
            input_path (str): Path to the raw input image.
            output_path (str): Path to save the processed JPG image.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.rgb_image = None
        self.inverted_image = None
        self.processed_image = None
        
        # Default RGB slope parameters (will be adjusted by sliders)
        self.r_left_slope = 1.0
        self.r_right_slope = 1.0
        self.g_left_slope = 1.0
        self.g_right_slope = 1.0
        self.b_left_slope = 1.0
        self.b_right_slope = 1.0
        
        # Default exposure and contrast values
        self.exposure = 0.0  # Exposure adjustment (0.0 = no change)
        self.contrast = 0.0  # Contrast adjustment (0.0 = no change)
        
        # Transformation state
        self.rotation_angle = 0  # In 90-degree increments (0, 90, 180, 270)
        self.is_flipped = False  # Horizontal flip state
        self.zoom_factor = 0.0   # Border cleanup zoom factor (0.0 = no zoom, 0.1 = 10% zoom)

    def load_raw_image(self, input_path=None):
        """Load a raw image into memory"""
        if input_path:
            self.input_path = input_path
            
        if not self.input_path:
            return None
            
        try:
            with rawpy.imread(self.input_path) as raw:
                self.rgb_image = raw.postprocess(output_bps=8, use_camera_wb=True)
            self.inverted_image = 255 - self.rgb_image
            
            # Reset transformation state when loading a new image
            self.rotation_angle = 0
            self.is_flipped = False
            
            return self.inverted_image
        except Exception as e:
            print(f"Error loading raw image: {e}")
            return None

    def apply_triangle_tone_curves(self, rgb_image=None, 
                                  r_left_slope=None, r_right_slope=None, 
                                  g_left_slope=None, g_right_slope=None, 
                                  b_left_slope=None, b_right_slope=None, 
                                  exposure=None, contrast=None):
        """
        Apply triangle-based tone curves and additional adjustments to an input RGB image.

        Parameters:
            rgb_image (numpy.ndarray): The input RGB image.
            r_left_slope (float): Left slope adjustment for red channel.
            r_right_slope (float): Right slope adjustment for red channel.
            g_left_slope (float): Left slope adjustment for green channel.
            g_right_slope (float): Right slope adjustment for green channel.
            b_left_slope (float): Left slope adjustment for blue channel.
            b_right_slope (float): Right slope adjustment for blue channel.
            exposure (float): Exposure adjustment value (-1.0 to 1.0).
            contrast (float): Contrast adjustment value (-1.0 to 1.0).

        Returns:
            numpy.ndarray: The transformed image after applying adjustments.
            tuple: The tone curves for each channel (for visualization).
        """
        if rgb_image is None:
            if self.inverted_image is None:
                return None, None
            rgb_image = self.inverted_image
            
        # Update parameters if provided
        if r_left_slope is not None:
            self.r_left_slope = r_left_slope
        if r_right_slope is not None:
            self.r_right_slope = r_right_slope
        if g_left_slope is not None:
            self.g_left_slope = g_left_slope
        if g_right_slope is not None:
            self.g_right_slope = g_right_slope
        if b_left_slope is not None:
            self.b_left_slope = b_left_slope
        if b_right_slope is not None:
            self.b_right_slope = b_right_slope
        if exposure is not None:
            self.exposure = exposure
        if contrast is not None:
            self.contrast = contrast
            
        left_slopes = [self.r_left_slope, self.g_left_slope, self.b_left_slope]
        right_slopes = [self.r_right_slope, self.g_right_slope, self.b_right_slope]
        colors = ('r', 'g', 'b')
        
        tone_curves = []
        transformed_channels = []

        def narrow_dominant_triangle(hist, threshold=0.05):
            # Make sure histogram has values
            if hist.size == 0 or np.max(hist) == 0:
                # Return default values to avoid empty sequence errors
                return 0, 128, 255

            peak_idx = np.argmax(hist)
            peak_value = hist[peak_idx]
            
            # Safely find left boundary
            left_mask = hist[:peak_idx] >= threshold * peak_value
            left_idx = 0  # Default if no values found
            if np.any(left_mask):
                left_idx = np.argmax(left_mask)
            
            # Safely find right boundary
            right_mask = hist[peak_idx:] < threshold * peak_value
            right_idx = 255  # Default if no values found
            if np.any(right_mask):
                right_idx = peak_idx + np.argmax(right_mask)
                
            return left_idx, peak_idx, right_idx

        def create_tone_curve(left, right, left_slope, right_slope, length=256):
            # Create base curve
            curve = np.zeros(length, dtype=np.uint8)
            
            # Get the midpoint of the curve
            midpoint = (left + right) // 2
            
            # Apply left slope adjustment
            if left_slope != 1.0:
                # Calculate new left point based on left_slope
                # For slopes > 1: moves right (steeper)
                # For slopes < 1: moves left (flatter)
                new_left = int(midpoint - (midpoint - left) * (1.0 / left_slope))
                
                # Make sure adjusted point is within valid range
                new_left = max(0, new_left)
                
                left = new_left
            
            # Apply right slope adjustment
            if right_slope != 1.0:
                # Calculate new right point based on right_slope
                # For slopes > 1: moves left (steeper)
                # For slopes < 1: moves right (flatter)
                new_right = int(midpoint + (right - midpoint) * (1.0 / right_slope))
                
                # Make sure adjusted point is within valid range
                new_right = min(length - 1, new_right)
                
                right = new_right
            
            # Generate the curve
            if left < right:
                curve[left:right + 1] = np.linspace(0, 255, right - left + 1)
                curve[right + 1:] = 255
            else:
                # Fallback for invalid cases
                curve = np.linspace(0, 255, length, dtype=np.uint8)
            
            return curve

        # Apply triangle-based tone curves
        for i, color in enumerate(colors):
            try:
                hist = cv2.calcHist([rgb_image], [i], None, [256], [0, 256]).flatten()
                left, peak, right = narrow_dominant_triangle(hist)

                # Apply the current slopes for this channel
                tone_curve = create_tone_curve(left, right, left_slopes[i], right_slopes[i])
                tone_curves.append(tone_curve)

                channel = rgb_image[:, :, i]
                transformed_channel = cv2.LUT(channel, tone_curve)
                transformed_channels.append(transformed_channel)
            except Exception as e:
                print(f"Error processing {color} channel: {str(e)}")
                # Create a default linear tone curve and apply it
                default_curve = np.linspace(0, 255, 256, dtype=np.uint8)
                tone_curves.append(default_curve)
                channel = rgb_image[:, :, i]
                transformed_channel = cv2.LUT(channel, default_curve)
                transformed_channels.append(transformed_channel)

        # Merge channels after tone curve application
        transformed_image = cv2.merge(transformed_channels)
        
        # Apply exposure adjustment
        if self.exposure != 0.0:
            # Convert to float for processing
            img_float = transformed_image.astype(np.float32)
            
            # Apply exposure (values > 0 brighten, values < 0 darken)
            if self.exposure > 0:
                # For brightening: scale from 0 to 1.0 (max 2x brighter)
                factor = 1.0 + self.exposure
                img_float = img_float * factor
            else:
                # For darkening: scale from 0 to -1.0 (max 0.5x darker)
                factor = 1.0 + self.exposure
                img_float = img_float * factor
            
            # Clip values to valid range
            img_float = np.clip(img_float, 0, 255)
            
            # Convert back to uint8
            transformed_image = img_float.astype(np.uint8)
        
        # Apply contrast adjustment
        if self.contrast != 0.0:
            # Convert to float for processing
            img_float = transformed_image.astype(np.float32)
            
            # Calculate the midpoint (128 for 8-bit images)
            midpoint = 128
            
            # Apply contrast adjustment
            if self.contrast > 0:
                # Increase contrast: stretch values away from midpoint
                factor = 1.0 + self.contrast
                img_float = (img_float - midpoint) * factor + midpoint
            else:
                # Decrease contrast: compress values toward midpoint
                factor = 1.0 + self.contrast  # Note: contrast is negative here
                img_float = (img_float - midpoint) * factor + midpoint
            
            # Clip values to valid range
            img_float = np.clip(img_float, 0, 255)
            
            # Convert back to uint8
            transformed_image = img_float.astype(np.uint8)
        
        self.processed_image = transformed_image
        
        # Apply any stored transformations
        self.apply_transformations()
        
        return self.processed_image, tone_curves
        
    def apply_transformations(self):
        """Apply stored rotation and flip transformations to the processed image"""
        if self.processed_image is None:
            return
            
        # Apply rotation if needed
        if self.rotation_angle > 0:
            # Apply rotation based on the stored angle
            if self.rotation_angle == 90:
                self.processed_image = cv2.rotate(self.processed_image, cv2.ROTATE_90_CLOCKWISE)
            elif self.rotation_angle == 180:
                self.processed_image = cv2.rotate(self.processed_image, cv2.ROTATE_180)
            elif self.rotation_angle == 270:
                self.processed_image = cv2.rotate(self.processed_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Apply flip if needed
        if self.is_flipped:
            self.processed_image = cv2.flip(self.processed_image, 1)  # 1 for horizontal flip
            
        # Apply zoom factor if needed
        if self.zoom_factor > 0.0:
            self.apply_zoom()

    def apply_zoom(self):
        """Apply zoom to processed image based on zoom_factor"""
        if self.processed_image is None or self.zoom_factor <= 0.0:
            return
            
        height, width = self.processed_image.shape[:2]
        
        # Calculate crop margin in pixels
        margin_x = int(width * self.zoom_factor)
        margin_y = int(height * self.zoom_factor)
        
        # Apply crop
        if margin_x > 0 and margin_y > 0 and margin_x < width/2 and margin_y < height/2:
            self.processed_image = self.processed_image[margin_y:height-margin_y, margin_x:width-margin_x]

    def rotate_image(self):
        """Rotate the processed image 90 degrees clockwise and update rotation state"""
        if self.processed_image is None:
            return None
        
        # Update rotation angle (0, 90, 180, 270)
        self.rotation_angle = (self.rotation_angle + 90) % 360
        
        # Rotate 90 degrees clockwise
        rotated = cv2.rotate(self.processed_image, cv2.ROTATE_90_CLOCKWISE)
        self.processed_image = rotated
        return rotated
    
    def flip_image(self):
        """Flip the processed image horizontally and update flip state"""
        if self.processed_image is None:
            return None
        
        # Toggle flip state
        self.is_flipped = not self.is_flipped
        
        # Flip horizontally (across the y-axis)
        flipped = cv2.flip(self.processed_image, 1)  # 1 means horizontal flip
        self.processed_image = flipped
        return flipped
        
    def reset_transformations(self):
        """Reset all transformations and reapply tone curves"""
        self.rotation_angle = 0
        self.is_flipped = False
        self.zoom_factor = 0.0
        
        # Reapply the tone curves to get back to untransformed state
        if self.inverted_image is not None:
            self.apply_triangle_tone_curves()
        
        return self.processed_image

    def add_white_border(self, image=None, border_size=100):
        """
        Add a white border around the image.
        
        Parameters:
            image (numpy.ndarray): The image to add border to
            border_size (int): Size of the border in pixels
            
        Returns:
            numpy.ndarray: The image with white border added
        """
        if image is None:
            image = self.processed_image
            
        if image is None:
            return None
            
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Create a new white canvas with border
        new_height = height + 2 * border_size
        new_width = width + 2 * border_size
        
        # Create a white canvas (255 for all RGB channels)
        bordered_image = np.ones((new_height, new_width, 3), dtype=np.uint8) * 255
        
        # Place the original image in the center
        bordered_image[border_size:border_size+height, border_size:border_size+width] = image
        
        return bordered_image
        
    def save_as_jpg(self, output_path=None, image=None, add_border=False, border_size=100):
        """
        Save the provided RGB image as a JPG file.

        Parameters:
            output_path (str): Path to save the image to.
            image (numpy.ndarray): The RGB image to save.
            add_border (bool): Whether to add a white border.
            border_size (int): Size of the border in pixels if add_border is True.
        """
        if output_path:
            self.output_path = output_path
            
        if image is None:
            image = self.processed_image
            
        if image is None or self.output_path is None:
            return False
            
        try:
            # Add white border if requested
            if add_border:
                image = self.add_white_border(image, border_size)
            
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(self.output_path, bgr_image)
            return True
        except Exception as e:
            print(f"Error saving image: {e}")
            return False


class HistogramCanvas(FigureCanvas):
    """Canvas for displaying histograms and tone curves with separate plots for each channel"""
    
    def __init__(self, parent=None, width=5, height=6, dpi=100):
        # Create figure with transparent background
        self.fig = plt.figure(figsize=(width, height), dpi=dpi, facecolor='none')
        self.fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98, hspace=0.1)
        
        # Create axes in a grid - 3 plots (RGB channels)
        self.axes = [self.fig.add_subplot(3, 1, i+1) for i in range(3)]
        
        # Make sure they share x-axis
        for i in range(1, 3):
            self.axes[i].sharex(self.axes[0])
            
        super(HistogramCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        # Create twin axes for tone curves
        self.twin_axes = [ax.twinx() for ax in self.axes]
        
        # Configure for extremely minimalist appearance
        for ax in self.axes:
            ax.set_facecolor('none')
            ax.set_title("")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.grid(False)
        
        for ax in self.twin_axes:
            ax.set_facecolor('none')
            ax.set_ylabel("")
            ax.tick_params(right=False, labelright=False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_ylim(0, 255)
        
        # Configure channel colors
        self.channels = [('Red', 'r'), ('Green', 'g'), ('Blue', 'b')]
        
        # Store original image for comparison
        self.original_image = None
        
        # Set minimum size to prevent negative dimensions during resize
        self.setMinimumSize(200, 300)
        
    def resizeEvent(self, event):
        """Override to handle resizing more safely"""
        # Only update layout if we have a valid size
        width, height = self.get_width_height()
        if width > 0 and height > 0:
            try:
                self.fig.tight_layout(pad=0.1)
            except:
                pass  # Ignore layout errors during resize
        super(HistogramCanvas, self).resizeEvent(event)

    def plot_histogram_and_curves(self, original_image, tone_curves, processed_image=None):
        """
        Plot RGB histograms and tone curves
        
        Parameters:
            original_image: The original (inverted) image
            tone_curves: The tone curves to display
            processed_image: The processed image after all adjustments
        """
        if original_image is None or tone_curves is None:
            return
            
        # Store original image for later comparisons
        self.original_image = original_image
        
        # Clear all axes
        for ax in self.axes:
            ax.clear()
        for ax in self.twin_axes:
            ax.clear()
        
        # Reset minimalist style
        for ax in self.axes:
            ax.set_title("")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.grid(False)
        
        for ax in self.twin_axes:
            ax.set_ylabel("")
            ax.tick_params(right=False, labelright=False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
        
        x = np.arange(0, 256)
        
        # Plot RGB histograms and tone curves
        for i, (name, color) in enumerate(self.channels):
            # Plot histogram on the left y-axis with respective color
            hist = cv2.calcHist([original_image], [i], None, [256], [0, 256])
            # Normalize histogram for better visibility
            hist_max = hist.max()
            if hist_max > 0:
                hist = hist / hist_max * 0.9  # Scale to 90% of plot height
            self.axes[i].fill_between(x, hist.flatten(), alpha=0.2, color=color)
            self.axes[i].plot(hist, color=color, alpha=0.7, linewidth=1.0)
            
            # Plot tone curve on the right y-axis with light gray color
            if tone_curves and i < len(tone_curves):
                self.twin_axes[i].plot(x, tone_curves[i], color='#aaaaaa', linewidth=1.0)
            
            # Set x-limits
            self.axes[i].set_xlim([0, 255])
        
        # Set y-limits for tone curves
        for ax in self.twin_axes:
            ax.set_ylim([0, 255])
        
        # Update layout safely
        try:
            self.fig.tight_layout(pad=0.1)
        except:
            pass  # Ignore layout errors
            
        self.draw()


class ImageViewer(QLabel):
    """Widget for displaying the image"""
    
    def __init__(self, parent=None):
        super(ImageViewer, self).__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #222222;")  # Darker background
        self.setFrameShape(QFrame.NoFrame)  # Remove frame
        self.original_pixmap = None
        
    def set_image(self, image):
        """Set the image to display from a numpy array"""
        if image is None:
            self.clear()
            self.original_pixmap = None
            return
            
        # Convert numpy array to QImage
        height, width, channels = image.shape
        bytes_per_line = channels * width
        
        # Create a copy of the array to ensure it's contiguous in memory
        # and use numpy's tobytes() method instead of accessing .data directly
        image_copy = np.ascontiguousarray(image)
        q_image = QImage(image_copy.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Store the original pixmap
        self.original_pixmap = QPixmap.fromImage(q_image)
        
        # Scale to fit the label while preserving aspect ratio
        self.setPixmap(self.original_pixmap.scaled(
            self.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        ))
        
    def resizeEvent(self, event):
        """Handle resize events to scale the image"""
        if self.original_pixmap and not self.original_pixmap.isNull():
            self.setPixmap(self.original_pixmap.scaled(
                self.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            ))
        super(ImageViewer, self).resizeEvent(event)


class ToneCurveEditor(QMainWindow):
    """Main window for the Orpheus Negative Lab application"""
    
    def __init__(self):
        super(ToneCurveEditor, self).__init__()
        
        self.processor = ToneCurveProcessor()
        self.current_directory = None
        self.image_files = []
        self.current_index = -1
        
        # Debounce timer for sliders
        self.slider_timer = QTimer()
        self.slider_timer.setSingleShot(True)
        self.slider_timer.timeout.connect(self.update_image_after_slider)
        
        # Store slider values to apply after debounce
        self.pending_slider_values = None
        
        # Dictionary to store image-specific settings
        self.image_settings = {}
        
        # Key tracking for keyboard shortcuts
        self.keys_pressed = set()
        
        # Keyboard shortcuts enabled flag (off by default)
        self.keyboard_shortcuts_enabled = False
        
        # Load Roboto font if available
        self.load_roboto_font()
        
        self.init_ui()
        
    def load_roboto_font(self):
        """Load Roboto font for the application"""
        # Try to find Roboto on the system
        font_db = QFontDatabase()
        
        # Check if Roboto is already installed in the system
        roboto_families = [f for f in font_db.families() if 'roboto' in f.lower()]
        
        # If Roboto isn't found, we'll use system defaults but specify sans-serif as fallback
        self.font_available = len(roboto_families) > 0
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('Orpheus Negative Lab')
        self.setMinimumSize(1000, 800)
        
        # Font family setting based on availability
        font_family = "'Roboto', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"
        
        # Apply minimalist application style
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: #1a1a1a;
                color: #e0e0e0;
                font-family: {font_family};
            }}
            QLabel {{
                color: #e0e0e0;
                font-family: {font_family};
            }}
            QPushButton {{
                background-color: #2a2a2a;
                color: #e0e0e0;
                border: none;
                padding: 6px;
                border-radius: 2px;
                font-family: {font_family};
                font-weight: normal;
            }}
            QPushButton:hover {{
                background-color: #3a3a3a;
            }}
            QPushButton:pressed {{
                background-color: #444444;
            }}
            QSlider {{
                height: 20px;
            }}
            QSlider::groove:horizontal {{
                height: 3px;
                background: #444444;
            }}
            QSlider::handle:horizontal {{
                background: #888888;
                width: 12px;
                margin: -5px 0;
                border-radius: 6px;
            }}
            QSlider::handle:horizontal:hover {{
                background: #aaaaaa;
            }}
            QFrame {{
                border: none;
            }}
            QSplitter::handle {{
                background-color: #333333;
            }}
            QStatusBar {{
                color: #888888;
                font-family: {font_family};
                font-size: 11px;
            }}
            QMenuBar {{
                background-color: #1a1a1a;
                color: #e0e0e0;
                font-family: {font_family};
            }}
            QMenuBar::item {{
                background-color: transparent;
            }}
            QMenuBar::item:selected {{
                background-color: #3a3a3a;
            }}
            QMenu {{
                background-color: #1a1a1a;
                color: #e0e0e0;
                font-family: {font_family};
            }}
            QMenu::item:selected {{
                background-color: #3a3a3a;
            }}
            QMessageBox {{
                background-color: #1a1a1a;
                color: #e0e0e0;
                font-family: {font_family};
            }}
        """)
        
        # Make main window focusable to capture keyboard events
        self.setFocusPolicy(Qt.StrongFocus)
        
        # Create status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        
        # Create central widget and main layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)  # Reduced padding
        main_layout.setSpacing(8)  # Reduced spacing
        
        # Create splitter for image view and controls
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)  # Thinner splitter handle
        
        # Left panel: Image viewer
        self.image_viewer = ImageViewer()
        splitter.addWidget(self.image_viewer)
        
        # Right panel: Controls and histogram
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)  # Reduced padding
        right_layout.setSpacing(8)  # Reduced spacing
        
        # Histogram display
        histogram_frame = QFrame()
        histogram_frame.setFrameShape(QFrame.NoFrame)  # Remove frame
        histogram_layout = QVBoxLayout(histogram_frame)
        histogram_layout.setContentsMargins(0, 0, 0, 0)  # No margins for minimalist look
        
        # Create histogram (RGB channels only)
        self.histogram_canvas = HistogramCanvas(histogram_frame, width=5, height=6)
        histogram_layout.addWidget(self.histogram_canvas)
        
        right_layout.addWidget(histogram_frame, stretch=3)
        
        # Sliders groups
        sliders_group = QWidget()
        sliders_layout = QGridLayout(sliders_group)
        sliders_layout.setContentsMargins(0, 0, 0, 0)  # No margins
        sliders_layout.setSpacing(4)  # Tighter spacing
        
        # Create sliders
        self.create_slider_group(sliders_layout)
        right_layout.addWidget(sliders_group, stretch=2)
        
        # Image transformation buttons
        transform_group = QWidget()
        transform_layout = QHBoxLayout(transform_group)
        transform_layout.setContentsMargins(0, 0, 0, 0)  # No margins
        transform_layout.setSpacing(6)  # Reduced spacing
        
        self.rotate_button = QPushButton("Rotate 90°")
        self.rotate_button.clicked.connect(self.rotate_current_image)
        self.rotate_button.setMinimumHeight(30)
        
        self.flip_button = QPushButton("Flip Horizontal")
        self.flip_button.clicked.connect(self.flip_current_image)
        self.flip_button.setMinimumHeight(30)
        
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_current_image)
        self.reset_button.setMinimumHeight(30)
        
        transform_layout.addWidget(self.rotate_button)
        transform_layout.addWidget(self.flip_button)
        transform_layout.addWidget(self.reset_button)
        
        right_layout.addWidget(transform_group)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(6)  # Reduced spacing
        
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.load_previous_image)
        self.prev_button.setMinimumHeight(30)
        
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.load_next_image)
        self.next_button.setMinimumHeight(30)
        
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_current_image)
        self.save_button.setMinimumHeight(30)
        
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        nav_layout.addWidget(self.save_button)
        
        right_layout.addLayout(nav_layout)
        
        # Add keyboard shortcuts info label
        self.shortcuts_label = QLabel("Keyboard Shortcuts: Disabled")
        self.shortcuts_label.setStyleSheet("color: #666666; font-size: 10px;")  # More subtle text
        self.shortcuts_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.shortcuts_label)
        
        # Add right panel to splitter
        splitter.addWidget(right_panel)
        
        # Set initial sizes
        splitter.setSizes([600, 400])
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
        
        # Set central widget
        self.setCentralWidget(central_widget)
        
        # Add menu bar
        self.create_menu()
        
        # Update the UI state
        self.update_ui_state()
        
    def create_slider_group(self, layout):
        """Create all adjustment sliders with labels"""
        # Title for tone curve sliders section
        tone_curve_label = QLabel("Tone Curve Adjustments")
        tone_curve_label.setStyleSheet("font-size: 12px; color: #aaaaaa; font-weight: medium;")  # Medium weight for header
        layout.addWidget(tone_curve_label, 0, 0, 1, 4)
        
        # Red channel sliders
        red_label = QLabel("Red:")
        red_label.setStyleSheet("color: #ff8080;")  # Reddish tint
        layout.addWidget(red_label, 1, 0)
        
        red_left_label = QLabel("Left:")
        self.red_left_slider = QSlider(Qt.Horizontal)
        self.red_left_slider.setRange(1, 200)  # 0.5x to 2.0x
        self.red_left_slider.setValue(100)      # Default is 1.0x
        self.red_left_slider.setTickPosition(QSlider.NoTicks)  # Remove ticks for cleaner look
        self.red_left_value_label = QLabel("1.0")
        self.red_left_slider.valueChanged.connect(self.slider_changed)
        
        layout.addWidget(red_left_label, 1, 1)
        layout.addWidget(self.red_left_slider, 1, 2)
        layout.addWidget(self.red_left_value_label, 1, 3)
        
        red_right_label = QLabel("Right:")
        self.red_right_slider = QSlider(Qt.Horizontal)
        self.red_right_slider.setRange(1, 200)
        self.red_right_slider.setValue(100)
        self.red_right_slider.setTickPosition(QSlider.NoTicks)  # Remove ticks
        self.red_right_value_label = QLabel("1.0")
        self.red_right_slider.valueChanged.connect(self.slider_changed)
        
        layout.addWidget(red_right_label, 2, 1)
        layout.addWidget(self.red_right_slider, 2, 2)
        layout.addWidget(self.red_right_value_label, 2, 3)
        
        # Green channel sliders
        green_label = QLabel("Green:")
        green_label.setStyleSheet("color: #80ff80;")  # Greenish tint
        layout.addWidget(green_label, 3, 0)
        
        green_left_label = QLabel("Left:")
        self.green_left_slider = QSlider(Qt.Horizontal)
        self.green_left_slider.setRange(1, 200)
        self.green_left_slider.setValue(100)
        self.green_left_slider.setTickPosition(QSlider.NoTicks)
        self.green_left_value_label = QLabel("1.0")
        self.green_left_slider.valueChanged.connect(self.slider_changed)
        
        layout.addWidget(green_left_label, 3, 1)
        layout.addWidget(self.green_left_slider, 3, 2)
        layout.addWidget(self.green_left_value_label, 3, 3)
        
        green_right_label = QLabel("Right:")
        self.green_right_slider = QSlider(Qt.Horizontal)
        self.green_right_slider.setRange(1, 200)
        self.green_right_slider.setValue(100)
        self.green_right_slider.setTickPosition(QSlider.NoTicks)
        self.green_right_value_label = QLabel("1.0")
        self.green_right_slider.valueChanged.connect(self.slider_changed)
        
        layout.addWidget(green_right_label, 4, 1)
        layout.addWidget(self.green_right_slider, 4, 2)
        layout.addWidget(self.green_right_value_label, 4, 3)
        
        # Blue channel sliders
        blue_label = QLabel("Blue:")
        blue_label.setStyleSheet("color: #8080ff;")  # Blueish tint
        layout.addWidget(blue_label, 5, 0)
        
        blue_left_label = QLabel("Left:")
        self.blue_left_slider = QSlider(Qt.Horizontal)
        self.blue_left_slider.setRange(1, 200)
        self.blue_left_slider.setValue(100)
        self.blue_left_slider.setTickPosition(QSlider.NoTicks)
        self.blue_left_value_label = QLabel("1.0")
        self.blue_left_slider.valueChanged.connect(self.slider_changed)
        
        layout.addWidget(blue_left_label, 5, 1)
        layout.addWidget(self.blue_left_slider, 5, 2)
        layout.addWidget(self.blue_left_value_label, 5, 3)
        
        blue_right_label = QLabel("Right:")
        self.blue_right_slider = QSlider(Qt.Horizontal)
        self.blue_right_slider.setRange(1, 200)
        self.blue_right_slider.setValue(100)
        self.blue_right_slider.setTickPosition(QSlider.NoTicks)
        self.blue_right_value_label = QLabel("1.0")
        self.blue_right_slider.valueChanged.connect(self.slider_changed)
        
        layout.addWidget(blue_right_label, 6, 1)
        layout.addWidget(self.blue_right_slider, 6, 2)
        layout.addWidget(self.blue_right_value_label, 6, 3)
        
        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Plain)
        separator.setStyleSheet("background-color: #333333; max-height: 1px;")  # Thin subtle line
        layout.addWidget(separator, 7, 0, 1, 4)
        
        # Title for additional adjustments section
        additional_label = QLabel("Additional Adjustments")
        additional_label.setStyleSheet("font-size: 12px; color: #aaaaaa;")  # Subtle header
        layout.addWidget(additional_label, 8, 0, 1, 4)
        
        # Exposure slider
        exposure_label = QLabel("Exposure:")
        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setRange(-100, 100)  # -1.0 to +1.0
        self.exposure_slider.setValue(0)         # Default is 0 (no change)
        self.exposure_slider.setTickPosition(QSlider.NoTicks)
        self.exposure_value_label = QLabel("0.0")
        self.exposure_slider.valueChanged.connect(self.slider_changed)
        
        layout.addWidget(exposure_label, 9, 0)
        layout.addWidget(self.exposure_slider, 9, 1, 1, 2)
        layout.addWidget(self.exposure_value_label, 9, 3)
        
        # Contrast slider
        contrast_label = QLabel("Contrast:")
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(-100, 100)  # -1.0 to +1.0
        self.contrast_slider.setValue(0)         # Default is 0 (no change)
        self.contrast_slider.setTickPosition(QSlider.NoTicks)
        self.contrast_value_label = QLabel("0.0")
        self.contrast_slider.valueChanged.connect(self.slider_changed)
        
        layout.addWidget(contrast_label, 10, 0)
        layout.addWidget(self.contrast_slider, 10, 1, 1, 2)
        layout.addWidget(self.contrast_value_label, 10, 3)
        
        # Crop slider
        crop_label = QLabel("Crop:")
        self.crop_slider = QSlider(Qt.Horizontal)
        self.crop_slider.setRange(0, 100)  # 0-10% in 0.1% increments
        self.crop_slider.setValue(0)
        self.crop_slider.setTickPosition(QSlider.NoTicks)
        self.crop_value_label = QLabel("0.0%")
        self.crop_slider.valueChanged.connect(self.crop_slider_changed)
        
        layout.addWidget(crop_label, 11, 0)
        layout.addWidget(self.crop_slider, 11, 1, 1, 2)
        layout.addWidget(self.crop_value_label, 11, 3)
        
        # Create horizontal layout for buttons
        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget)
        button_layout.setContentsMargins(0, 0, 0, 0)  # No margins
        button_layout.setSpacing(6)  # Small spacing between buttons
        
        # Auto-crop button
        self.auto_crop_button = QPushButton("Auto-crop (5%)")
        self.auto_crop_button.clicked.connect(self.apply_auto_crop)
        self.auto_crop_button.setStyleSheet("background-color: #2a2a2a; padding: 4px;")  # Flatten button
        
        # Add Auto-Warm button
        self.auto_warm_button = QPushButton("Auto-Warm")
        self.auto_warm_button.clicked.connect(self.apply_auto_warm)
        self.auto_warm_button.setStyleSheet("background-color: #2a2a2a; padding: 4px;")  # Flatten button
        
        # Add buttons to the horizontal layout
        button_layout.addWidget(self.auto_crop_button)
        button_layout.addWidget(self.auto_warm_button)
        
        # Add the button container to the main grid layout
        layout.addWidget(button_widget, 12, 1, 1, 2)
    
    def get_current_image_key(self):
        """Get a unique key for the current image for settings storage"""
        if self.current_index < 0 or self.current_index >= len(self.image_files):
            return None
        return str(self.image_files[self.current_index])
    
    def save_current_image_settings(self):
        """Save the current image settings to the dictionary"""
        key = self.get_current_image_key()
        if not key:
            return
        
        # Store all relevant settings
        self.image_settings[key] = {
            'r_left_slope': self.red_left_slider.value() / 100.0,
            'r_right_slope': self.red_right_slider.value() / 100.0,
            'g_left_slope': self.green_left_slider.value() / 100.0,
            'g_right_slope': self.green_right_slider.value() / 100.0,
            'b_left_slope': self.blue_left_slider.value() / 100.0,
            'b_right_slope': self.blue_right_slider.value() / 100.0,
            'exposure': self.exposure_slider.value() / 100.0,
            'contrast': self.contrast_slider.value() / 100.0,
            'rotation': self.processor.rotation_angle,
            'flipped': self.processor.is_flipped,
            'zoom_factor': self.processor.zoom_factor
        }
    
    def load_image_settings(self, key):
        """Load image settings if they exist"""
        if key in self.image_settings:
            settings = self.image_settings[key]
            
            # Apply stored slider values
            if 'r_left_slope' in settings:
                self.red_left_slider.setValue(int(settings['r_left_slope'] * 100))
            if 'r_right_slope' in settings:
                self.red_right_slider.setValue(int(settings['r_right_slope'] * 100))
            if 'g_left_slope' in settings:
                self.green_left_slider.setValue(int(settings['g_left_slope'] * 100))
            if 'g_right_slope' in settings:
                self.green_right_slider.setValue(int(settings['g_right_slope'] * 100))
            if 'b_left_slope' in settings:
                self.blue_left_slider.setValue(int(settings['b_left_slope'] * 100))
            if 'b_right_slope' in settings:
                self.blue_right_slider.setValue(int(settings['b_right_slope'] * 100))
            
            # Apply stored exposure and contrast values if they exist
            if 'exposure' in settings:
                self.exposure_slider.setValue(int(settings['exposure'] * 100))
            if 'contrast' in settings:
                self.contrast_slider.setValue(int(settings['contrast'] * 100))
            
            # Update processor transformation state
            self.processor.rotation_angle = settings['rotation']
            self.processor.is_flipped = settings['flipped']
            
            # Apply zoom factor if it exists
            if 'zoom_factor' in settings:
                self.processor.zoom_factor = settings['zoom_factor']
                # Update the crop slider to match the stored value
                self.crop_slider.setValue(int(settings['zoom_factor'] * 1000))
            
            return True
        
        return False
        
    def reset_sliders(self):
        """Reset all sliders to default values"""
        self.red_left_slider.setValue(100)
        self.red_right_slider.setValue(100)
        self.green_left_slider.setValue(100)
        self.green_right_slider.setValue(100)
        self.blue_left_slider.setValue(100)
        self.blue_right_slider.setValue(100)
        self.exposure_slider.setValue(0)
        self.contrast_slider.setValue(0)
        self.crop_slider.setValue(0)
        
    def slider_changed(self):
        """
        Handle slider value changes with debounce.
        Updates tone curves in real-time while debouncing the image update.
        """
        # Update slider value labels immediately
        self.red_left_value_label.setText(f"{self.red_left_slider.value() / 100.0:.1f}")
        self.red_right_value_label.setText(f"{self.red_right_slider.value() / 100.0:.1f}")
        self.green_left_value_label.setText(f"{self.green_left_slider.value() / 100.0:.1f}")
        self.green_right_value_label.setText(f"{self.green_right_slider.value() / 100.0:.1f}")
        self.blue_left_value_label.setText(f"{self.blue_left_slider.value() / 100.0:.1f}")
        self.blue_right_value_label.setText(f"{self.blue_right_slider.value() / 100.0:.1f}")
        self.exposure_value_label.setText(f"{self.exposure_slider.value() / 100.0:.2f}")
        self.contrast_value_label.setText(f"{self.contrast_slider.value() / 100.0:.2f}")
        
        # Get current slider values
        r_left_slope = self.red_left_slider.value() / 100.0
        r_right_slope = self.red_right_slider.value() / 100.0
        g_left_slope = self.green_left_slider.value() / 100.0
        g_right_slope = self.green_right_slider.value() / 100.0
        b_left_slope = self.blue_left_slider.value() / 100.0
        b_right_slope = self.blue_right_slider.value() / 100.0
        exposure = self.exposure_slider.value() / 100.0
        contrast = self.contrast_slider.value() / 100.0
        
        # Store the values for later image update (after debounce)
        self.pending_slider_values = {
            'r_left': r_left_slope,
            'r_right': r_right_slope,
            'g_left': g_left_slope,
            'g_right': g_right_slope,
            'b_left': b_left_slope,
            'b_right': b_right_slope,
            'exposure': exposure,
            'contrast': contrast
        }
        
        # Create tone curves with current values (without applying to image)
        if self.processor.inverted_image is not None:
            # Update the processor's values (without updating the processed image)
            self.processor.r_left_slope = r_left_slope
            self.processor.r_right_slope = r_right_slope
            self.processor.g_left_slope = g_left_slope
            self.processor.g_right_slope = g_right_slope
            self.processor.b_left_slope = b_left_slope
            self.processor.b_right_slope = b_right_slope
            self.processor.exposure = exposure
            self.processor.contrast = contrast
            
            # Calculate tone curves for visualization only
            tone_curves = self.calculate_tone_curves_only(self.processor.inverted_image)
            
            # Update the histogram with the new tone curves (without updating the image)
            self.histogram_canvas.plot_histogram_and_curves(self.processor.inverted_image, tone_curves)
        
        # Restart the debounce timer for the actual image processing
        self.slider_timer.start(150)  # 150ms debounce
    
    def calculate_tone_curves_only(self, rgb_image):
        """
        Calculate tone curves without updating the processed image.
        Used for real-time visualization during slider movement.
        """
        if rgb_image is None:
            return None
        
        left_slopes = [self.processor.r_left_slope, self.processor.g_left_slope, self.processor.b_left_slope]
        right_slopes = [self.processor.r_right_slope, self.processor.g_right_slope, self.processor.b_right_slope]
        colors = ('r', 'g', 'b')
        tone_curves = []
        
        def narrow_dominant_triangle(hist, threshold=0.05):
            # Make sure histogram has values
            if hist.size == 0 or np.max(hist) == 0:
                # Return default values to avoid empty sequence errors
                return 0, 128, 255

            peak_idx = np.argmax(hist)
            peak_value = hist[peak_idx]
            
            # Safely find left boundary
            left_mask = hist[:peak_idx] >= threshold * peak_value
            left_idx = 0  # Default if no values found
            if np.any(left_mask):
                left_idx = np.argmax(left_mask)
            
            # Safely find right boundary
            right_mask = hist[peak_idx:] < threshold * peak_value
            right_idx = 255  # Default if no values found
            if np.any(right_mask):
                right_idx = peak_idx + np.argmax(right_mask)
                
            return left_idx, peak_idx, right_idx
        
        def create_tone_curve(left, right, left_slope, right_slope, length=256):
            # Create base curve
            curve = np.zeros(length, dtype=np.uint8)
            
            # Get the midpoint of the curve
            midpoint = (left + right) // 2
            
            # Apply left slope adjustment
            if left_slope != 1.0:
                # Calculate new left point based on left_slope
                # For slopes > 1: moves right (steeper)
                # For slopes < 1: moves left (flatter)
                new_left = int(midpoint - (midpoint - left) * (1.0 / left_slope))
                
                # Make sure adjusted point is within valid range
                new_left = max(0, new_left)
                
                left = new_left
            
            # Apply right slope adjustment
            if right_slope != 1.0:
                # Calculate new right point based on right_slope
                # For slopes > 1: moves left (steeper)
                # For slopes < 1: moves right (flatter)
                new_right = int(midpoint + (right - midpoint) * (1.0 / right_slope))
                
                # Make sure adjusted point is within valid range
                new_right = min(length - 1, new_right)
                
                right = new_right
            
            # Generate the curve
            if left < right:
                curve[left:right + 1] = np.linspace(0, 255, right - left + 1)
                curve[right + 1:] = 255
            else:
                # Fallback for invalid cases
                curve = np.linspace(0, 255, length, dtype=np.uint8)
            
            return curve
        
        for i, color in enumerate(colors):
            try:
                hist = cv2.calcHist([rgb_image], [i], None, [256], [0, 256]).flatten()
                left, peak, right = narrow_dominant_triangle(hist)
                tone_curve = create_tone_curve(left, right, left_slopes[i], right_slopes[i])
                tone_curves.append(tone_curve)
            except Exception as e:
                print(f"Error processing {color} channel: {str(e)}")
                # Create a default linear tone curve on error
                default_curve = np.linspace(0, 255, 256, dtype=np.uint8)
                tone_curves.append(default_curve)
        
        return tone_curves
    
    def update_image_after_slider(self):
        """Update the image after the slider debounce period"""
        if self.pending_slider_values and self.processor.inverted_image is not None:
            r_left_slope = self.pending_slider_values['r_left']
            r_right_slope = self.pending_slider_values['r_right']
            g_left_slope = self.pending_slider_values['g_left']
            g_right_slope = self.pending_slider_values['g_right']
            b_left_slope = self.pending_slider_values['b_left']
            b_right_slope = self.pending_slider_values['b_right']
            exposure = self.pending_slider_values['exposure']
            contrast = self.pending_slider_values['contrast']
            
            # Apply tone curves to create the processed image
            processed_image, tone_curves = self.processor.apply_triangle_tone_curves(
                None, 
                r_left_slope, r_right_slope,
                g_left_slope, g_right_slope,
                b_left_slope, b_right_slope,
                exposure, contrast
            )
            
            # Update the image display
            self.image_viewer.set_image(processed_image)
            
            # Update histogram with both original and processed image
            self.histogram_canvas.plot_histogram_and_curves(
                self.processor.inverted_image, 
                tone_curves,
                processed_image
            )
            
            # Save the current settings
            self.save_current_image_settings()
    
    def load_current_image(self):
        """Load and display the current image"""
        if self.current_index < 0 or self.current_index >= len(self.image_files):
            return
            
        raw_file = self.image_files[self.current_index]
        image_key = str(raw_file)
        
        try:
            # Load the image
            inverted_image = self.processor.load_raw_image(str(raw_file))
            
            if inverted_image is not None:
                # Try to load saved settings or use defaults
                settings_loaded = self.load_image_settings(image_key)
                
                if not settings_loaded:
                    # Reset to defaults if no saved settings
                    self.reset_sliders()
                    self.processor.reset_transformations()
                
                # Apply tone curves with current settings
                r_left_slope = self.red_left_slider.value() / 100.0
                r_right_slope = self.red_right_slider.value() / 100.0
                g_left_slope = self.green_left_slider.value() / 100.0
                g_right_slope = self.green_right_slider.value() / 100.0
                b_left_slope = self.blue_left_slider.value() / 100.0
                b_right_slope = self.blue_right_slider.value() / 100.0
                exposure = self.exposure_slider.value() / 100.0
                contrast = self.contrast_slider.value() / 100.0
                
                processed_image, tone_curves = self.processor.apply_triangle_tone_curves(
                    inverted_image, 
                    r_left_slope, r_right_slope,
                    g_left_slope, g_right_slope,
                    b_left_slope, b_right_slope,
                    exposure, contrast
                )
                
                # Display image and histogram
                self.image_viewer.set_image(processed_image)
                self.histogram_canvas.plot_histogram_and_curves(
                    inverted_image, 
                    tone_curves,
                    processed_image
                )
                
                # Update window title
                self.setWindowTitle(f'Orpheus Negative Lab - {raw_file.name} ({self.current_index + 1}/{len(self.image_files)})')
                
        except Exception as e:
            print(f"Error loading image: {e}")
            
        self.update_ui_state()
    
    def load_next_image(self):
        """Load the next image in the directory"""
        if self.current_index < len(self.image_files) - 1:
            # Save current image settings before moving
            self.save_current_image_settings()
            
            self.current_index += 1
            self.load_current_image()
            
    def load_previous_image(self):
        """Load the previous image in the directory"""
        if self.current_index > 0:
            # Save current image settings before moving
            self.save_current_image_settings()
            
            self.current_index -= 1
            self.load_current_image()
    
    def rotate_current_image(self):
        """Rotate the current image 90 degrees clockwise"""
        if self.processor.processed_image is not None:
            rotated_image = self.processor.rotate_image()
            self.image_viewer.set_image(rotated_image)
            
            # Save the updated settings
            self.save_current_image_settings()
    
    def flip_current_image(self):
        """Flip the current image horizontally"""
        if self.processor.processed_image is not None:
            flipped_image = self.processor.flip_image()
            self.image_viewer.set_image(flipped_image)
            
            # Save the updated settings
            self.save_current_image_settings()
    
    def reset_current_image(self):
        """Reset all transformations and sliders for current image"""
        if self.processor.inverted_image is not None:
            # Reset sliders to default
            self.reset_sliders()
            
            # Reset transformations
            processed_image = self.processor.reset_transformations()
            
            # Update display
            self.image_viewer.set_image(processed_image)
            
            # Re-apply default tone curves
            r_left_slope = self.red_left_slider.value() / 100.0
            r_right_slope = self.red_right_slider.value() / 100.0
            g_left_slope = self.green_left_slider.value() / 100.0
            g_right_slope = self.green_right_slider.value() / 100.0
            b_left_slope = self.blue_left_slider.value() / 100.0
            b_right_slope = self.blue_right_slider.value() / 100.0
            exposure = self.exposure_slider.value() / 100.0
            contrast = self.contrast_slider.value() / 100.0
            
            processed_image, tone_curves = self.processor.apply_triangle_tone_curves(
                None, 
                r_left_slope, r_right_slope,
                g_left_slope, g_right_slope,
                b_left_slope, b_right_slope,
                exposure, contrast
            )
            
            # Update histogram
            self.histogram_canvas.plot_histogram_and_curves(self.processor.inverted_image, tone_curves)
            
            # Remove saved settings for this image
            key = self.get_current_image_key()
            if key and key in self.image_settings:
                del self.image_settings[key]

    def update_ui_state(self):
        """Update the enabled/disabled state of UI elements"""
        has_images = len(self.image_files) > 0
        has_current = self.current_index >= 0 and self.current_index < len(self.image_files)
        has_processed = has_current and self.processor.processed_image is not None
        
        # Update navigation buttons
        self.prev_button.setEnabled(has_current and self.current_index > 0)
        self.next_button.setEnabled(has_current and self.current_index < len(self.image_files) - 1)
        self.save_button.setEnabled(has_processed)
        
        # Update transformation buttons
        self.rotate_button.setEnabled(has_processed)
        self.flip_button.setEnabled(has_processed)
        self.reset_button.setEnabled(has_processed)
        self.crop_slider.setEnabled(has_processed)
        self.auto_crop_button.setEnabled(has_processed)
        self.auto_warm_button.setEnabled(has_processed)  # Enable/disable Auto-Warm button
        
        # Update all sliders
        self.red_left_slider.setEnabled(has_current)
        self.red_right_slider.setEnabled(has_current)
        self.green_left_slider.setEnabled(has_current)
        self.green_right_slider.setEnabled(has_current)
        self.blue_left_slider.setEnabled(has_current)
        self.blue_right_slider.setEnabled(has_current)
        self.exposure_slider.setEnabled(has_current)
        self.contrast_slider.setEnabled(has_current)

    def create_menu(self):
        """Create the application menu"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        open_action = file_menu.addAction('Open Directory...')
        open_action.triggered.connect(self.open_directory)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction('Exit')
        exit_action.triggered.connect(self.close)
        
        # Tools menu
        tools_menu = menubar.addMenu('Tools')
        
        save_action = tools_menu.addAction('Save Current Image')
        save_action.triggered.connect(self.save_current_image)
        
        save_all_action = tools_menu.addAction('Process All Images')
        save_all_action.triggered.connect(self.process_all_images)
        
        tools_menu.addSeparator()
        
        # Add keyboard shortcuts toggle action
        self.toggle_shortcuts_action = QAction('Enable Keyboard Shortcuts', self)
        self.toggle_shortcuts_action.setCheckable(True)
        self.toggle_shortcuts_action.setChecked(False)
        self.toggle_shortcuts_action.triggered.connect(self.toggle_keyboard_shortcuts)
        tools_menu.addAction(self.toggle_shortcuts_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        user_guide_action = help_menu.addAction('User Guide')
        user_guide_action.triggered.connect(self.show_user_guide)
        
        about_action = help_menu.addAction('About')
        about_action.triggered.connect(self.show_about)
        
    def open_directory(self):
        """Open a directory containing RAW images"""
        directory = QFileDialog.getExistingDirectory(
            self, 
            "Select Directory with RAW Images",
            ""
        )
        
        if directory:
            self.current_directory = directory
            self.load_images_from_directory()
            
    def load_images_from_directory(self):
        """Load all supported RAW files from the current directory"""
        if not self.current_directory:
            return
            
        # Get all files with supported extensions
        supported_extensions = ['.cr2', '.cr3']
        raw_files = []
        
        for ext in supported_extensions:
            raw_files.extend(list(Path(self.current_directory).glob(f'*{ext}')))
            raw_files.extend(list(Path(self.current_directory).glob(f'*{ext.upper()}')))
        
        self.image_files = sorted(raw_files)
        
        if self.image_files:
            self.current_index = 0
            self.load_current_image()
        else:
            self.current_index = -1
            self.update_ui_state()
            QApplication.beep()
            
    def save_current_image(self):
        """Save the current processed image with a white border"""
        if self.current_index < 0 or self.processor.processed_image is None:
            return
            
        # Create output directory if it doesn't exist
        output_dir = os.path.join(self.current_directory, 'new_positives')
        os.makedirs(output_dir, exist_ok=True)
        
        # Create output path
        raw_file = self.image_files[self.current_index]
        output_filename = raw_file.stem + ".jpg"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save the image with a 50px white border
        if self.processor.save_as_jpg(output_path, add_border=True, border_size=100):
            self.statusBar.showMessage(f"Image saved with white border to: {output_path}", 3000)
            print(f"Image saved to: {output_path}")
            
    def process_all_images(self):
        """Process all images in the directory with current settings"""
        if not self.image_files:
            return
            
        # Create output directory
        output_dir = os.path.join(self.current_directory, 'new_positives')
        os.makedirs(output_dir, exist_ok=True)
        
        # Get current slider values
        r_left_slope = self.red_left_slider.value() / 100.0
        r_right_slope = self.red_right_slider.value() / 100.0
        g_left_slope = self.green_left_slider.value() / 100.0
        g_right_slope = self.green_right_slider.value() / 100.0
        b_left_slope = self.blue_left_slider.value() / 100.0
        b_right_slope = self.blue_right_slider.value() / 100.0
        exposure = self.exposure_slider.value() / 100.0
        contrast = self.contrast_slider.value() / 100.0
        
        # Process each file
        successful = 0
        failed = 0
        total_images = len(self.image_files)
        
        # Show initial progress in status bar
        self.statusBar.showMessage(f"Processing images: 0/{total_images} completed...")
        QApplication.processEvents()  # Update the UI
        
        for i, raw_file in enumerate(self.image_files):
            try:
                # Update status bar with current progress
                self.statusBar.showMessage(f"Processing images: {i}/{total_images} - Current: {raw_file.name}")
                QApplication.processEvents()  # Force UI update
                
                print(f"Processing ({i+1}/{total_images}): {raw_file.name}")
                
                # Create output filename
                output_filename = raw_file.stem + ".jpg"
                output_path = os.path.join(output_dir, output_filename)
                
                # Create a new processor for each image
                processor = ToneCurveProcessor(str(raw_file), output_path)
                
                # Load and process the image
                inverted_image = processor.load_raw_image()
                processed_image, _ = processor.apply_triangle_tone_curves(
                    inverted_image, 
                    r_left_slope, r_right_slope,
                    g_left_slope, g_right_slope,
                    b_left_slope, b_right_slope,
                    exposure, contrast
                )
                
                # Always apply 5% crop for batch processing
                processor.zoom_factor = 0.05  # 5% crop
                processed_image, _ = processor.apply_triangle_tone_curves()
                
                # Save with white border
                processor.save_as_jpg(add_border=True, border_size=100)
                
                successful += 1
                
                # Update status bar with success count
                self.statusBar.showMessage(f"Processing images: {i+1}/{total_images} - Completed: {successful}, Failed: {failed}")
                QApplication.processEvents()  # Force UI update
                
            except Exception as e:
                failed += 1
                print(f"Error processing {raw_file}: {str(e)}")
                
                # Update status bar with error info
                self.statusBar.showMessage(f"Processing images: {i+1}/{total_images} - Error with {raw_file.name}")
                QApplication.processEvents()  # Force UI update
        
        # Print summary
        print(f"\nProcessing complete! Successfully processed: {successful}, Failed: {failed}")
        self.statusBar.showMessage(f"Processing complete! Processed {successful} images with 5% crop and white borders. Failed: {failed}", 5000)

    def show_user_guide(self):
        """Display the user guide for the application"""
        guide_text = """
<html>
<head>
<style>
    body { font-family: 'Roboto', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif; color: #e0e0e0; }
    h2 { font-weight: 300; color: #ffffff; }
    h3 { font-weight: 400; color: #cccccc; }
    ul { margin-left: 15px; }
    li { margin-bottom: 4px; }
</style>
</head>
<body>
<h2>Orpheus Negative Lab - User Guide</h2>

<h3>Getting Started</h3>
<p>1. Open a directory containing RAW image files (.CR2, .CR3) using <b>File > Open Directory</b>.</p>
<p>2. Navigate through images using the <b>Previous</b> and <b>Next</b> buttons.</p>

<h3>Image Adjustments</h3>
<p><b>Tone Curve Adjustments:</b></p>
<ul>
    <li>Adjust the slope sliders for each RGB channel to modify tonal response</li>
    <li>Left Slope: controls the shadow to midtone response</li>
    <li>Right Slope: controls the midtone to highlight response</li>
</ul>

<p><b>Additional Adjustments:</b></p>
<ul>
    <li>Exposure: Adjust overall brightness (-1.0 to +1.0)</li>
    <li>Contrast: Adjust image contrast (-1.0 to +1.0)</li>
    <li>Crop: Adjust crop percentage (0-10%) to remove unwanted border artifacts</li>
    <li>Auto-crop (5%): Automatically applies a 5% crop to the image edges</li>
</ul>

<h3>Image Transformations</h3>
<ul>
    <li><b>Rotate 90°</b>: Rotates the image 90 degrees clockwise</li>
    <li><b>Flip Horizontal</b>: Mirrors the image horizontally</li>
    <li><b>Reset</b>: Resets all transformations and adjustments</li>
</ul>

<h3>Keyboard Shortcuts</h3>
<p>Keyboard shortcuts are disabled by default. Enable them using the "Enable Keyboard Shortcuts" button in the toolbar.</p>
<ul>
    <li><b>r + Left/Right Arrow</b>: Adjust the Red left slope</li>
    <li><b>Shift+r + Left/Right Arrow</b>: Adjust the Red right slope</li>
    <li><b>g + Left/Right Arrow</b>: Adjust the Green left slope</li>
    <li><b>Shift+g + Left/Right Arrow</b>: Adjust the Green right slope</li>
    <li><b>b + Left/Right Arrow</b>: Adjust the Blue left slope</li>
    <li><b>Shift+b + Left/Right Arrow</b>: Adjust the Blue right slope</li>
    <li><b>e + Left/Right Arrow</b>: Adjust Exposure</li>
    <li><b>c + Left/Right Arrow</b>: Adjust Contrast</li>
    <li><b>Cmd/Ctrl + Left/Right Arrow</b>: Navigate between images</li>
</ul>

<h3>Saving Images</h3>
<ul>
    <li><b>Save</b>: Saves the current image as a JPG with a 50px white border</li>
    <li><b>Process All Images</b>: Processes all images with the current settings and adds white borders</li>
</ul>

<p>All processed images are saved to a 'new_positives' folder in the current directory.</p>
</body>
</html>
"""
        
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("User Guide")
        msg_box.setTextFormat(Qt.RichText)
        msg_box.setText(guide_text)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()
        
    def show_about(self):
        """Display information about the application"""
        about_text = """
<html>
<head>
<style>
    body { font-family: 'Roboto', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif; color: #e0e0e0; }
    h2 { font-weight: 300; color: #ffffff; }
    p { margin: 8px 0; }
</style>
</head>
<body>
<h2>Orpheus Negative Lab</h2>
<p>Version 1.0</p>
<p>A powerful tool for processing RAW images with advanced tone curve adjustments.</p>
<p>Designed for creating inverted negative film images with precise control.</p>
</body>
</html>
"""
        
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("About")
        msg_box.setTextFormat(Qt.RichText)
        msg_box.setText(about_text)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

    def toggle_keyboard_shortcuts(self):
        """Toggle keyboard shortcuts on/off"""
        self.keyboard_shortcuts_enabled = not self.keyboard_shortcuts_enabled
        
        # Update the menu item checkbox state
        self.toggle_shortcuts_action.setChecked(self.keyboard_shortcuts_enabled)
        
        # Update shortcuts information label
        if self.keyboard_shortcuts_enabled:
            self.shortcuts_label.setText("Keyboard Shortcuts: r/g/b + ←→ for adjustments, e/c + ←→ for exposure/contrast, ⌘←→ to navigate")
            self.statusBar.showMessage("Keyboard shortcuts enabled", 3000)
        else:
            self.shortcuts_label.setText("Keyboard Shortcuts: Disabled")
            self.statusBar.showMessage("Keyboard shortcuts disabled", 3000)

    def crop_slider_changed(self):
        """Handle changes to the crop slider value"""
        # Convert the slider value to a percentage (0-10%)
        zoom_percent = self.crop_slider.value() / 10.0
        self.crop_value_label.setText(f"{zoom_percent:.1f}%")
        
        # Update the processor's zoom factor
        if self.processor.inverted_image is not None:
            self.processor.zoom_factor = zoom_percent / 100.0  # Convert to decimal (0.0-0.1)
            
            # Reapply transformations with the new zoom factor
            processed_image, tone_curves = self.processor.apply_triangle_tone_curves()
            
            # Update the image display
            self.image_viewer.set_image(processed_image)
            
            # Save the current settings
            self.save_current_image_settings()
    
    def apply_auto_crop(self):
        """Apply automatic 5% crop to the current image"""
        if self.processor.inverted_image is not None:
            # Set crop slider to 5%
            self.crop_slider.setValue(50)  # 50/10 = 5%
            
            # This will trigger crop_slider_changed which will update the image
            self.statusBar.showMessage("Applied 5% auto-crop", 3000)
            
    def apply_auto_warm(self):
        """Apply auto-warm filter by changing slider values"""
        if self.processor.inverted_image is not None:
            # Apply warming effect according to specifications:
            # 1. Right red: 20% higher than default (120)
            # 2. Right green: 10% lower than default (90)
            # 3. Left blue: 30% higher than default (130)
            
            # Set default = 100 and calculate new values
            default_value = 100
            red_right_value = int(default_value * 1.2)  # 20% higher
            green_right_value = int(default_value * 0.9)  # 10% lower
            blue_left_value = int(default_value * 1.3)  # 30% higher
            
            # Apply new values to sliders
            self.red_right_slider.setValue(red_right_value)
            self.green_right_slider.setValue(green_right_value)
            self.blue_left_slider.setValue(blue_left_value)
            
            # The slider_changed function will handle updating the image
            self.statusBar.showMessage("Applied warming filter", 3000)


def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    window = ToneCurveEditor()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 