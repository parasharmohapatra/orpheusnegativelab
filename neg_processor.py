'''
AUTHOR: Parashar Mohapatra
'''
import rawpy
import numpy as np
import os
from numba import njit, prange
import time
import jax.numpy as jnp
from jax import jit
from jax import lax
from scipy.special import expit
import cv2

class NegativeImageProcessor:
    def __init__(self):
        self.original_rgb = None
        self.current_rgb = None
        self.original_rgb_copy = None

    def open_image(self, image_data):
        # Initialize target values for color balance
        target_blue_yellow = 128  # Target value for blue-yellow balance (middle of LAB b channel)
        target_magenta_green = 124  # Target value for magenta-green balance (middle of LAB a channel)
        max_iterations = 10  # Maximum iterations for convergence
        tolerance = 1.0  # Acceptable difference from target
        
        self.original_rgb = image_data
        self.current_rgb = image_data.copy()
        self.original_rgb_copy = 65535 - self.current_rgb
        #self.original_rgb_copy = self.auto_contrast_adjust(self.original_rgb_copy)
        # Initial white balance
        self.original_rgb_copy = self.gray_world_white_balance(self.original_rgb_copy)
        self.original_rgb_copy = self.auto_balance_green_magenta_numpy(self.original_rgb_copy)
        
        # Convert to 8-bit for LAB conversion
        img_8bit = cv2.normalize(self.original_rgb_copy, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Initialize adjustment factors
        tint_adj_factor = 1.0
        white_balance_adj_factor = 1.0
        blacks_adj = 0
        shadows_adj = 0
        highlights_adj = 0
        whites_adj = 0
        
        # Iterative convergence for color balance
        for _ in range(max_iterations):
            # Convert to LAB color space
            lab_image = cv2.cvtColor(img_8bit, cv2.COLOR_RGB2LAB)
            
            # Calculate current mean values
            mean_a = np.mean(lab_image[:, :, 1])  # a channel (magenta-green)
            mean_b = np.mean(lab_image[:, :, 2])  # b channel (blue-yellow)
            
            # Check if we've converged for color balance
            if (abs(mean_a - target_magenta_green) < tolerance and
                abs(mean_b - target_blue_yellow) < tolerance):
                break
            
            # Adjust factors based on difference from target
            tint_adj_factor *= (target_magenta_green / mean_a) ** 0.5
            white_balance_adj_factor *= (target_blue_yellow / mean_b) ** 0.5
            
            # Apply adjustments
            adjusted_image = self.adjust_tint_16bit(self.original_rgb_copy, tint_adj_factor)
            adjusted_image = self.adjust_white_balance_blue_yellow_16bit(adjusted_image, white_balance_adj_factor)
            
            # Update for next iteration
            img_8bit = cv2.normalize(adjusted_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Apply final color balance adjustments to the image
        self.original_rgb_copy = self.adjust_tint_16bit(self.original_rgb_copy, tint_adj_factor)
        self.original_rgb_copy = self.adjust_white_balance_blue_yellow_16bit(self.original_rgb_copy, white_balance_adj_factor)
        #self.original_rgb_copy = self.auto_contrast_adjust(self.original_rgb_copy)


        # Apply final tonal adjustments to the image
        self.original_rgb_copy = self.adjust_tones(self.original_rgb_copy,
                                                  blacks=blacks_adj,
                                                  shadows=shadows_adj,
                                                  highlights=highlights_adj,
                                                  whites=whites_adj)
        return tint_adj_factor, white_balance_adj_factor, blacks_adj, shadows_adj, highlights_adj, whites_adj

    def find_clipped_corners(self, hist, bins, clip_percentage):
        total_pixels = np.sum(hist)
        clip_pixels = total_pixels * clip_percentage

        cumulative = np.cumsum(hist)
        left_idx = np.searchsorted(cumulative, clip_pixels)
        right_idx = np.searchsorted(cumulative, total_pixels - clip_pixels)
        right_idx = np.searchsorted(cumulative, total_pixels - clip_pixels)

        left_idx = max(0, min(left_idx, len(bins) - 2))
        right_idx = max(0, min(right_idx, len(bins) - 2))

        left = bins[:-1][left_idx]
        right = bins[:-1][right_idx]
        return left, right

    def create_tone_curve(self, left, right, adj_factor=0):
        curve = np.zeros(65536, dtype=np.uint16)

        if adj_factor == 0:
            curve[:int(left)] = 0
            curve[int(right):] = 65535
            if int(right) > int(left):
                curve[int(left):int(right)] = np.linspace(
                    0, 65535, int(right) - int(left), endpoint=False, dtype=np.uint16
                )
            return curve

        mid = (left + right) / 2
        y_mid = 65535 - ((mid - left) / (right - left)) * 65535
        adjusted_y_mid = y_mid + adj_factor * 65535

        curve[:int(left)] = 0
        curve[int(right):] = 65535
        if int(right) > int(left):
            curve[int(left):int(mid)] = np.linspace(
                0, adjusted_y_mid, int(mid) - int(left), endpoint=False, dtype=np.uint16
            )
            curve[int(mid):int(right)] = np.linspace(
                adjusted_y_mid, 65535, int(right) - int(mid), endpoint=False, dtype=np.uint16
            )

        return curve
    
    def create_tone_curve_s_curve(self, left, right, adj_factor=0):
        curve = np.zeros(65536, dtype=np.uint16)
    
        # Define the S-curve parameters
        mid = (left + right) / 2
        width = right - left
    
        # Generate the S-curve using the sigmoid function
        x = np.linspace(0, 65535, 65536)
        s_curve = expit((x - mid) / (width / 2))  # Adjust steepness with width/8
    
        # Scale the S-curve to the range [0, 65535]
        s_curve = (s_curve - s_curve.min()) / (s_curve.max() - s_curve.min())
        s_curve = 65535 - (s_curve * 65535).astype(np.uint16)  # Invert the curve
    
        # Apply adjustment factor
        if adj_factor != 0:
            adjusted_mid = mid + adj_factor * width
            s_curve = expit((x - adjusted_mid) / (width / 4))
            s_curve = (s_curve - s_curve.min()) / (s_curve.max() - s_curve.min())
            s_curve = 65535 - (s_curve * 65535).astype(np.uint16)
    
        curve = 65535/2 - (s_curve-65535/2)
        return curve
    
    @staticmethod
    @njit(parallel=True, fastmath=True)
    def create_tone_curve_s_curve_fast(left, right):
        curve = np.zeros(65536, dtype=np.uint16)
        mid = (left + right) / 2
        
        # Create lookup table for sigmoid function
        sigmoid_lut = np.empty(65536, dtype=np.float32)
        scale = 10.0  # Controls steepness of S-curve
        for i in range(65536):
            x = (i - left) / (right - left)
            if x <= 0:
                sigmoid_lut[i] = 0
            elif x >= 1:
                sigmoid_lut[i] = 1
            else:
                # Simplified sigmoid function
                x = (x - 0.5) * scale
                sigmoid_lut[i] = 1 / (1 + np.exp(-x))
        
        # Apply S-curve transformation
        for i in prange(65536):
            if i <= left:
                curve[i] = 0
            elif i >= right:
                curve[i] = 65535
            else:
                curve[i] = int(sigmoid_lut[i] * 65535)
        
        return curve
    
    @staticmethod
    @njit(parallel=True, fastmath=True)
    def gray_world_white_balance(image):
        # Pre-allocate output array and calculate means in one pass
        h, w = image.shape[:2]
        means = np.zeros(3, dtype=np.float32)
        
        # Calculate means using parallel reduction
        for c in range(3):
            sum_val = 0.0
            for i in prange(h):
                row_sum = 0.0
                for j in range(w):
                    row_sum += image[i, j, c]
                sum_val += row_sum
            means[c] = sum_val / (h * w)
        
        # Calculate scaling factors
        scale_r = means[1] / means[0]
        scale_b = means[1] / means[2]
        
        # Apply scaling using parallel processing
        balanced_img = np.empty_like(image)
        for i in prange(h):
            for j in range(w):
                # Copy green channel directly
                balanced_img[i, j, 1] = image[i, j, 1]
                # Scale red and blue channels with bounds checking
                val_r = image[i, j, 0] * scale_r
                val_b = image[i, j, 2] * scale_b
                balanced_img[i, j, 0] = min(max(val_r, 0), 65535)
                balanced_img[i, j, 2] = min(max(val_b, 0), 65535)
        
        return balanced_img

    def auto_balance_green_magenta_numpy(self, image):
        # Normalize the 16-bit image to 8-bit
        image_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Convert the image to LAB color space
        lab_image = cv2.cvtColor(image_8bit, cv2.COLOR_RGB2LAB)

        # Split the LAB image into its channels
        l, a, b = cv2.split(lab_image)

        # Calculate the mean of the A channel (green-magenta axis)
        mean_a = np.mean(a)

        # Shift the A channel to center it around 128 (neutral green-magenta)
        a = np.clip(a - (mean_a - 128), 0, 255).astype(np.uint8)

        # Merge the adjusted channels back
        balanced_lab = np.dstack((l, a, b))

        # Convert back to RGB color space
        balanced_image_8bit = cv2.cvtColor(balanced_lab, cv2.COLOR_LAB2RGB)

        # Convert the 8-bit balanced image back to 16-bit
        balanced_image_16bit = cv2.normalize(balanced_image_8bit, None, 0, 65535, cv2.NORM_MINMAX, dtype=cv2.CV_16U)

        return balanced_image_16bit
    
    def adjust_saturation_16bit(self, image, saturation_factor=1.0):
        # Normalize the 16-bit image to 8-bit
        image_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Convert the image from RGB to HLS (Hue, Lightness, Saturation)
        hls_image = cv2.cvtColor(image_8bit, cv2.COLOR_RGB2HLS)

        # Adjust the Saturation channel
        hls_image[:, :, 2] = np.clip(hls_image[:, :, 2] * saturation_factor, 0, 255)

        # Convert back to RGB
        adjusted_image_8bit = cv2.cvtColor(hls_image, cv2.COLOR_HLS2RGB)

        # Convert the 8-bit adjusted image back to 16-bit
        adjusted_image_16bit = cv2.normalize(adjusted_image_8bit, None, 0, 65535, cv2.NORM_MINMAX, dtype=cv2.CV_16U)

        return adjusted_image_16bit

    def adjust_tones(self, image, blacks=0, shadows=0, highlights=0, whites=0):
        # Normalize the 16-bit image to 8-bit
        image_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Convert the image to LAB color space
        lab_image = cv2.cvtColor(image_8bit, cv2.COLOR_RGB2LAB)

        # Split the LAB image into its channels
        l, a, b = cv2.split(lab_image)

        # Convert L channel to float for precise adjustments
        l = l.astype(np.float32)

        # Adjust blacks and shadows (lower luminance values)
        l = np.clip(l + blacks + shadows, 0, 255)

        # Adjust highlights and whites (higher luminance values)
        l = np.clip(l + highlights + whites, 0, 255)

        # Convert L channel back to uint8
        l = l.astype(np.uint8)

        # Merge the adjusted L channel back with A and B channels
        adjusted_lab = cv2.merge((l, a, b))

        # Convert back to RGB color space
        adjusted_image_8bit = cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2RGB)

        # Convert the 8-bit adjusted image back to 16-bit
        adjusted_image_16bit = cv2.normalize(adjusted_image_8bit, None, 0, 65535, cv2.NORM_MINMAX, dtype=cv2.CV_16U)

        return adjusted_image_16bit
    
    def adjust_white_balance_blue_yellow_16bit(self, image, blue_yellow_factor=1.0):
        # Normalize the 16-bit image to 8-bit
        image_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
        # Convert the image to LAB color space
        lab_image = cv2.cvtColor(image_8bit, cv2.COLOR_RGB2LAB)
    
        # Split the LAB image into its channels
        l, a, b = cv2.split(lab_image)
    
        # Adjust the B channel (blue-yellow axis)
        b = np.clip(b * blue_yellow_factor, 0, 255).astype(np.uint8)
    
        # Merge the adjusted channels back
        balanced_lab = cv2.merge((l, a, b))
    
        # Convert back to RGB color space
        balanced_image_8bit = cv2.cvtColor(balanced_lab, cv2.COLOR_LAB2RGB)
    
        # Convert the 8-bit balanced image back to 16-bit
        balanced_image_16bit = cv2.normalize(balanced_image_8bit, None, 0, 65535, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
    
        return balanced_image_16bit

    def adjust_tint_16bit(self, image, tint_factor=1.0):
        # Normalize the 16-bit image to 8-bit
        image_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Convert the image to LAB color space
        lab_image = cv2.cvtColor(image_8bit, cv2.COLOR_RGB2LAB)

        # Split the LAB image into its channels
        l, a, b = cv2.split(lab_image)

        # Adjust the A channel (green-magenta axis)
        a = np.clip(a * tint_factor, 0, 255).astype(np.uint8)

        # Merge the adjusted channels back
        tinted_lab = cv2.merge((l, a, b))

        # Convert back to RGB color space
        tinted_image_8bit = cv2.cvtColor(tinted_lab, cv2.COLOR_LAB2RGB)

        # Convert the 8-bit tinted image back to 16-bit
        tinted_image_16bit = cv2.normalize(tinted_image_8bit, None, 0, 65535, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
        return tinted_image_16bit

    def auto_contrast_adjust(self, image, clip_percentage):
        # Calculate the histogram and find the lower and upper percentiles for contrast stretching
        hist, bins = np.histogram(image.ravel(), bins=65536, range=(0, 65535))
        lower_bound, upper_bound = self.find_clipped_corners(hist, bins, clip_percentage)

        # Perform contrast stretching
        stretched = np.clip((image - lower_bound) * (65535 / (upper_bound - lower_bound)), 0, 65535)
        return stretched.astype(np.uint16)


    @staticmethod
    @jit
    def adjust_exposure(image, exposure_adj):
        factor = 2 ** exposure_adj
        max_val = 65535

        # Apply exposure adjustment
        adjusted = jnp.clip(image * factor, 0, max_val)
        return adjusted.astype(jnp.uint16)
    
    @staticmethod
    @jit
    def adjust_brightness(image, brightness_adj):
        max_val = 65535

        # Apply brightness adjustment
        adjusted = jnp.clip(image + (brightness_adj * max_val), 0, max_val)
        return adjusted.astype(jnp.uint16)
    
    @staticmethod
    @jit
    def adjust_contrast(image, contrast_adj):
        max_val = 65535
        midpoint = max_val / 2

        # Apply contrast adjustment
        adjusted = jnp.clip((image - midpoint) * contrast_adj + midpoint, 0, max_val)
        return adjusted.astype(jnp.uint16)

    @staticmethod
    @jit
    def adjust_gamma(image, gamma):
        inv_gamma = 1.0 / gamma
        max_val = 65535
    
        # Normalize, apply gamma correction, and scale back using lax operations
        adjusted = lax.pow(image / max_val, inv_gamma) * max_val
        return adjusted.astype(jnp.uint16)
    
    @staticmethod
    @jit
    def adjust_log(image, c):
        max_val = 65535  # Or adjust if using a different image type
        image = image.astype(jnp.float32) # Ensure we're working with floats for log
        # Add 1 to avoid log(0) errors and scale to a reasonable range
        log_image = c * jnp.log1p(image / max_val)  # jnp.log1p is numerically stable for small x

        adjusted = lax.floor(log_image * max_val).astype(jnp.uint16)

        return adjusted

    @staticmethod
    @jit
    def adjust_highlights(image, highlight_adj):
        max_val = 65535
        # Apply highlight adjustment
        adjusted = jnp.clip(image + (highlight_adj * (max_val - image)), 0, max_val)
        return adjusted.astype(jnp.uint16)

    @staticmethod
    @jit
    def adjust_shadows(image, shadow_adj):
        max_val = 65535
        # Apply shadow adjustment
        adjusted = jnp.clip(image + (shadow_adj * image), 0, max_val)
        return adjusted.astype(jnp.uint16)

    @staticmethod
    @njit(parallel=True)
    def apply_tone_curve(image, tone_curve):
        adjusted = np.zeros_like(image, dtype=np.uint16)
        for i in prange(image.shape[0]):
            for j in prange(image.shape[1]):
                adjusted[i, j] = tone_curve[image[i, j]]
        return adjusted

    def process_image(self, tint_adj_factor=1, white_balance_adj_factor=1, blacks=0, shadows=0, highlights=0, whites=0, gamma_adj=1.0, log_adj=1.0, clip_percentage=0.001):
        if self.original_rgb is None:
            raise ValueError("No image has been opened. Call open_image first.")

        rgb_to_process = self.original_rgb_copy.copy()
        r_hist, r_bins = np.histogram(rgb_to_process[..., 0].ravel(), bins=128, range=(0, 65535))
        g_hist, g_bins = np.histogram(rgb_to_process[..., 1].ravel(), bins=128, range=(0, 65535))
        b_hist, b_bins = np.histogram(rgb_to_process[..., 2].ravel(), bins=128, range=(0, 65535))

        r_left, r_right = self.find_clipped_corners(r_hist, r_bins, 0.01)
        g_left, g_right = self.find_clipped_corners(g_hist, g_bins, 0.01)
        b_left, b_right = self.find_clipped_corners(b_hist, b_bins, 0.01)

        r_curve = self.create_tone_curve_s_curve(r_left, r_right)
        g_curve = self.create_tone_curve_s_curve(g_left, g_right)
        b_curve = self.create_tone_curve_s_curve(b_left, b_right)

        adjusted_rgb = rgb_to_process.copy()
        adjusted_rgb[..., 0] = self.apply_tone_curve(rgb_to_process[..., 0], r_curve)
        adjusted_rgb[..., 1] = self.apply_tone_curve(rgb_to_process[..., 1], g_curve)
        adjusted_rgb[..., 2] = self.apply_tone_curve(rgb_to_process[..., 2], b_curve)

        adjusted_rgb = self.adjust_tint_16bit(adjusted_rgb, tint_adj_factor)
        adjusted_rgb = self.adjust_white_balance_blue_yellow_16bit(adjusted_rgb, white_balance_adj_factor)
        adjusted_rgb = self.auto_contrast_adjust(adjusted_rgb, clip_percentage)
        adjusted_rgb = self.adjust_tones(adjusted_rgb, blacks, shadows, highlights, whites)
        adjusted_rgb = self.adjust_gamma(adjusted_rgb, gamma_adj)
        adjusted_rgb = self.adjust_log(adjusted_rgb, log_adj)

        return adjusted_rgb

class ImageFileManager:
    def __init__(self, directory_path):
        if not os.path.isdir(directory_path):
            raise ValueError("Invalid directory path provided.")
        self.directory_path = directory_path
        self.file_paths = [
            os.path.join(self.directory_path, f)
            for f in os.listdir(self.directory_path)
            if os.path.isfile(os.path.join(self.directory_path, f))
        ]

    def get_file_path(self, index):
        return self.file_paths[index]

    def load_image(self, file_path):
        if not os.path.isfile(file_path):
            raise ValueError("Invalid file path provided.")
        with rawpy.imread(file_path) as raw:
            return raw.postprocess(
                output_bps=16, 
                use_camera_wb=True, 
                no_auto_bright=True, 
                gamma=(1, 1),
            )
