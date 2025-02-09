import rawpy
import numpy as np
import os
from numba import njit, prange
import time
import jax.numpy as jnp
from jax import jit
from jax import lax

class NegativeImageProcessor:
    def __init__(self, clip_percentage=0.001):
        self.clip_percentage = clip_percentage
        self.original_rgb = None
        self.current_rgb = None

    def open_image(self, image_data): # Accepts image data directly
        self.original_rgb = image_data
        self.current_rgb = image_data.copy()

    def find_clipped_corners(self, hist, bins):
        total_pixels = np.sum(hist)
        clip_pixels = total_pixels * self.clip_percentage

        cumulative = np.cumsum(hist)
        left_idx = np.searchsorted(cumulative, clip_pixels)
        right_idx = np.searchsorted(cumulative, total_pixels - clip_pixels)

        left_idx = max(0, min(left_idx, len(bins) - 2))
        right_idx = max(0, min(right_idx, len(bins) - 2))

        left = bins[:-1][left_idx]
        right = bins[:-1][right_idx]
        return left, right

    def create_tone_curve(self, left, right, adj_factor=0):
        curve = np.zeros(65536, dtype=np.uint16)

        if adj_factor == 0:
            curve[:int(left)] = 65535
            curve[int(right):] = 0
            if int(right) > int(left):
                curve[int(left):int(right)] = np.linspace(
                    65535, 0, int(right) - int(left), endpoint=False, dtype=np.uint16
                )
            return curve

        mid = (left + right) / 2
        y_mid = 65535 - ((mid - left) / (right - left)) * 65535
        adjusted_y_mid = y_mid + adj_factor * 65535

        curve[:int(left)] = 65535
        curve[int(right):] = 0
        if int(right) > int(left):
            curve[int(left):int(mid)] = np.linspace(
                65535, adjusted_y_mid, int(mid) - int(left), endpoint=False, dtype=np.uint16
            )
            curve[int(mid):int(right)] = np.linspace(
                adjusted_y_mid, 0, int(right) - int(mid), endpoint=False, dtype=np.uint16
            )

        return curve

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
    def adjust_gamma(image, gamma):
        inv_gamma = 1.0 / gamma
        max_val = 65535
    
        # Normalize, apply gamma correction, and scale back using lax operations
        adjusted = lax.pow(image / max_val, inv_gamma) * max_val
        return adjusted.astype(jnp.uint16)

    def process_image(self, r_adj_factor=0, g_adj_factor=0, b_adj_factor=0, gamma=1.0, exposure_adj=0):
        if self.original_rgb is None:
            raise ValueError("No image has been opened. Call open_image first.")

        rgb_to_process = self.original_rgb.copy()

        r_hist, r_bins = np.histogram(rgb_to_process[..., 0].ravel(), bins=128, range=(0, 65535))
        g_hist, g_bins = np.histogram(rgb_to_process[..., 1].ravel(), bins=128, range=(0, 65535))
        b_hist, b_bins = np.histogram(rgb_to_process[..., 2].ravel(), bins=128, range=(0, 65535))

        r_left, r_right = self.find_clipped_corners(r_hist, r_bins)
        g_left, g_right = self.find_clipped_corners(g_hist, g_bins)
        b_left, b_right = self.find_clipped_corners(b_hist, b_bins)

        r_curve = self.create_tone_curve(r_left, r_right, r_adj_factor)
        g_curve = self.create_tone_curve(g_left, g_right, g_adj_factor)
        b_curve = self.create_tone_curve(b_left, b_right, b_adj_factor)

        adjusted_rgb = rgb_to_process.copy()
        x = np.arange(65536, dtype=np.uint16)
        adjusted_rgb[..., 0] = np.interp(rgb_to_process[..., 0], x, r_curve).astype(np.uint16)
        adjusted_rgb[..., 1] = np.interp(rgb_to_process[..., 1], x, g_curve).astype(np.uint16)
        adjusted_rgb[..., 2] = np.interp(rgb_to_process[..., 2], x, b_curve).astype(np.uint16)

        adjusted_rgb = self.adjust_exposure(adjusted_rgb, exposure_adj)
        adjusted_rgb = self.adjust_gamma(adjusted_rgb, gamma)

        self.current_rgb = adjusted_rgb

        return adjusted_rgb


class NumbaOptimizedNegativeImageProcessor(NegativeImageProcessor):
    def __init__(self, clip_percentage=0.001):
        super().__init__(clip_percentage)

    @staticmethod
    @njit(parallel=True)
    def apply_tone_curve(image, tone_curve):
        adjusted = np.zeros_like(image, dtype=np.uint16)
        for i in prange(image.shape[0]):
            for j in prange(image.shape[1]):
                adjusted[i, j] = tone_curve[image[i, j]]
        return adjusted
    
    def process_image(self, r_adj_factor=0, g_adj_factor=0, b_adj_factor=0, gamma=1.0, exposure_adj=0):
        if self.original_rgb is None:
            raise ValueError("No image has been opened. Call open_image first.")

        rgb_to_process = self.original_rgb.copy()

        r_hist, r_bins = np.histogram(rgb_to_process[..., 0].ravel(), bins=128, range=(0, 65535))
        g_hist, g_bins = np.histogram(rgb_to_process[..., 1].ravel(), bins=128, range=(0, 65535))
        b_hist, b_bins = np.histogram(rgb_to_process[..., 2].ravel(), bins=128, range=(0, 65535))

        r_left, r_right = self.find_clipped_corners(r_hist, r_bins)
        g_left, g_right = self.find_clipped_corners(g_hist, g_bins)
        b_left, b_right = self.find_clipped_corners(b_hist, b_bins)

        r_curve = self.create_tone_curve(r_left, r_right, r_adj_factor)
        g_curve = self.create_tone_curve(g_left, g_right, g_adj_factor)
        b_curve = self.create_tone_curve(b_left, b_right, b_adj_factor)

        adjusted_rgb = rgb_to_process.copy()
        adjusted_rgb[..., 0] = self.apply_tone_curve(rgb_to_process[..., 0], r_curve)
        adjusted_rgb[..., 1] = self.apply_tone_curve(rgb_to_process[..., 1], g_curve)
        adjusted_rgb[..., 2] = self.apply_tone_curve(rgb_to_process[..., 2], b_curve)

        adjusted_rgb = self.adjust_exposure(adjusted_rgb, exposure_adj)
        adjusted_rgb = self.adjust_gamma(adjusted_rgb, gamma)

        self.current_rgb = adjusted_rgb

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
                output_bps=16, use_camera_wb=True, no_auto_bright=True, gamma=(1, 1)
            )