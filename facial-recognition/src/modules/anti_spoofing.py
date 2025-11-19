"""
Anti-Spoofing (Liveness Detection) Module
"""
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import logging
from src.utils.config import get_config

logger = logging.getLogger(__name__)

# Lazy import - only import deepface when actually needed
# This prevents TensorFlow/gRPC from loading at module import time
DEEPFACE_AVAILABLE = None

def _check_deepface_available():
    """Lazy check if deepface is available"""
    global DEEPFACE_AVAILABLE
    if DEEPFACE_AVAILABLE is None:
        try:
            from deepface import DeepFace
            DEEPFACE_AVAILABLE = True
        except ImportError:
            DEEPFACE_AVAILABLE = False
    return DEEPFACE_AVAILABLE

def _get_deepface():
    """Lazy import of DeepFace"""
    if _check_deepface_available():
        from deepface import DeepFace
        return DeepFace
    return None


class AntiSpoofingDetector:
    """
    Anti-spoofing detector for liveness detection
    Uses pre-trained models or heuristic methods
    """
    def __init__(self, method=None, device='cpu'):
        config = get_config()
        self.method = method if method is not None else config.anti_spoofing.method
        self.device = device
        self.config = config.anti_spoofing
        
        if self.method == 'deepface' and not _check_deepface_available():
            logger.warning("deepface not available. Falling back to enhanced_heuristic method.")
            self.method = 'enhanced_heuristic'
        
        if self.method == 'deepface':
            logger.info("Using DeepFace for anti-spoofing detection")
        elif self.method == 'enhanced_heuristic':
            logger.info("Using enhanced heuristic-based anti-spoofing detection")
        else:
            logger.info("Using basic heuristic-based anti-spoofing detection")
    
    def detect(self, image):
        """Detect if face is real or spoofed"""
        if self.method == 'deepface':
            return self._detect_deepface(image)
        elif self.method == 'enhanced_heuristic':
            return self._detect_enhanced_heuristic(image)
        else:
            return self._detect_heuristic(image)
    
    def _detect_deepface(self, image):
        """Use DeepFace for liveness detection"""
        try:
            # Lazy import DeepFace only when needed
            DeepFace = _get_deepface()
            if DeepFace is None:
                return self._detect_heuristic(image)
            
            # Convert PIL to numpy if needed
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image
            
            # DeepFace liveness detection
            # DeepFace may not have direct liveness detection
            # We'll use a combination of face analysis and heuristics
            result = DeepFace.analyze(
                img_path=img_array,
                actions=['age', 'gender'],
                enforce_detection=False,
                silent=True
            )
            
            # Simple heuristic: if face analysis works well, likely real
            # This is a simplified approach - in production, use dedicated liveness models
            confidence = 0.7  # Default confidence
            
            # Check image quality (real faces usually have better quality)
            config = get_config()
            anti_spoof_cfg = config.anti_spoofing
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Higher variance = sharper image = more likely real
            if laplacian_var > anti_spoof_cfg.sharpness_threshold_high:
                confidence = anti_spoof_cfg.confidence_high
            elif laplacian_var > anti_spoof_cfg.sharpness_threshold_medium:
                confidence = anti_spoof_cfg.confidence_medium
            else:
                confidence = anti_spoof_cfg.confidence_low
            
            is_real = confidence > anti_spoof_cfg.confidence_threshold
            
            return is_real, confidence
            
        except Exception as e:
            logger.error(f"DeepFace detection error: {e}")
            return self._detect_heuristic(image)
    
    def _detect_heuristic(self, image):
        """Heuristic-based liveness detection"""
        # Convert to numpy array
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Heuristic 1: Image sharpness (Laplacian variance)
        # Real faces usually have more detail/sharpness
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Heuristic 2: Texture analysis (Local Binary Pattern-like)
        # Real faces have more texture variation
        texture_score = self._compute_texture_score(gray)
        
        # Heuristic 3: Color distribution
        # Printed photos often have different color characteristics
        if len(img_array.shape) == 3:
            color_score = self._compute_color_score(img_array)
        else:
            color_score = 0.5
        
        config = get_config()
        anti_spoof_cfg = config.anti_spoofing
        
        # Combine heuristics (using legacy weights for backward compatibility)
        sharpness_score = min(laplacian_var / anti_spoof_cfg.sharpness_threshold_high, 1.0)  # Normalize to 0-1
        texture_score_norm = min(texture_score / 50.0, 1.0)  # Normalize to 0-1
        
        # Weighted combination using legacy config values
        confidence = (anti_spoof_cfg.legacy_sharpness_weight * sharpness_score + 
                     anti_spoof_cfg.legacy_texture_weight * texture_score_norm + 
                     anti_spoof_cfg.legacy_color_weight * color_score)
        
        # Threshold
        is_real = confidence > anti_spoof_cfg.confidence_threshold
        
        return is_real, float(confidence)
    
    def _compute_texture_score(self, gray):
        """Compute texture variation score"""
        # Compute gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Variance of gradient magnitude indicates texture
        texture_score = np.var(gradient_magnitude)
        
        return texture_score
    
    def _compute_color_score(self, img_array):
        """Compute color distribution score"""
        # Compute color variance in different channels
        color_var = np.var(img_array, axis=(0, 1))
        color_score = np.mean(color_var) / 100.0  # Normalize
        
        return min(color_score, 1.0)
    
    def _detect_enhanced_heuristic(self, image):
        """Enhanced heuristic-based liveness detection"""
        # Convert to numpy array
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        scores = {}
        
        # 1. Enhanced Sharpness Analysis
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / self.config.sharpness_threshold_high, 1.0)
        scores['sharpness'] = sharpness_score
        
        # 2. Enhanced Texture Analysis (LBP-like)
        if self.config.lbp_texture_enabled:
            texture_score = self._compute_enhanced_texture_score(gray)
        else:
            texture_score = self._compute_texture_score(gray) / 50.0
        scores['texture'] = min(texture_score, 1.0)
        
        # 3. Color Distribution
        if len(img_array.shape) == 3:
            color_score = self._compute_color_score(img_array)
        else:
            color_score = 0.5
        scores['color'] = color_score
        
        # 4. Frequency Domain Analysis (detects printed photos)
        if self.config.frequency_analysis_enabled:
            frequency_score = self._compute_frequency_score(gray)
            scores['frequency'] = frequency_score
        else:
            scores['frequency'] = 0.5
        
        # 5. Depth/Focus Analysis (real faces have depth variation)
        if self.config.depth_estimation_enabled:
            depth_score = self._compute_depth_score(gray)
            scores['depth'] = depth_score
        else:
            scores['depth'] = 0.5
        
        # 6. Motion Blur Detection (printed photos are static, real faces may have slight motion)
        if self.config.motion_blur_detection_enabled:
            motion_score = self._compute_motion_blur_score(gray)
            scores['motion'] = motion_score
        else:
            scores['motion'] = 0.5
        
        # Weighted combination using enhanced weights
        confidence = (
            self.config.sharpness_weight * scores['sharpness'] +
            self.config.texture_weight * scores['texture'] +
            self.config.color_weight * scores['color'] +
            self.config.frequency_weight * scores['frequency'] +
            self.config.depth_weight * scores['depth'] +
            self.config.motion_blur_weight * scores['motion']
        )
        
        # Threshold
        is_real = confidence > self.config.confidence_threshold
        
        return is_real, float(confidence)
    
    def _compute_enhanced_texture_score(self, gray):
        """
        Enhanced texture analysis using Local Binary Pattern (LBP) approach
        Real faces have more complex texture patterns than printed photos
        Optimized for performance using vectorized operations
        """
        # Resize if too large for performance
        h, w = gray.shape
        if h > 200 or w > 200:
            scale = min(200 / h, 200 / w)
            new_h, new_w = int(h * scale), int(w * scale)
            gray = cv2.resize(gray, (new_w, new_h))
            h, w = gray.shape
        
        # Compute LBP using vectorized operations for better performance
        # Extract 8-neighborhood using array slicing
        texture_map = np.zeros((h-2, w-2), dtype=np.uint8)
        
        # Vectorized LBP computation
        center = gray[1:h-1, 1:w-1]
        code = np.zeros_like(center, dtype=np.uint8)
        
        # 8 neighbors
        neighbors = [
            gray[0:h-2, 0:w-2],      # top-left
            gray[0:h-2, 1:w-1],      # top
            gray[0:h-2, 2:w],        # top-right
            gray[1:h-1, 2:w],        # right
            gray[2:h, 2:w],          # bottom-right
            gray[2:h, 1:w-1],        # bottom
            gray[2:h, 0:w-2],        # bottom-left
            gray[1:h-1, 0:w-2]       # left
        ]
        
        # Build LBP code bit by bit
        for idx, neighbor in enumerate(neighbors):
            code |= ((neighbor >= center).astype(np.uint8) << idx)
        
        texture_map = code
        
        # Compute histogram and entropy
        hist, _ = np.histogram(texture_map.flatten(), bins=256, range=(0, 256))
        hist = hist.astype(float)
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist /= hist_sum  # Normalize
        
        # Entropy measures texture complexity
        # Higher entropy = more complex texture = more likely real
        # Avoid log(0) by adding small epsilon
        hist_nonzero = hist[hist > 0]
        if len(hist_nonzero) > 0:
            entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))
            texture_score = entropy / 8.0  # Normalize (max entropy for 256 bins is ~8)
        else:
            texture_score = 0.0
        
        return min(texture_score, 1.0)
    
    def _compute_frequency_score(self, gray):
        """
        Frequency domain analysis
        Printed photos often have different frequency characteristics
        """
        # Apply FFT to analyze frequency content
        f_transform = np.fft.fft2(gray.astype(float))
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Normalize
        magnitude_spectrum = magnitude_spectrum / (magnitude_spectrum.max() + 1e-10)
        
        # Real faces typically have more high-frequency content
        # Printed photos may have artifacts in frequency domain
        h, w = gray.shape
        center_h, center_w = h // 2, w // 2
        
        # Analyze high-frequency content (edges, details)
        # Extract ring around center (high frequencies)
        y, x = np.ogrid[:h, :w]
        mask = ((x - center_w)**2 + (y - center_h)**2) > (min(h, w) * 0.3)**2
        high_freq_energy = np.mean(magnitude_spectrum[mask])
        
        # Real faces have more high-frequency energy
        frequency_score = min(high_freq_energy * 2.0, 1.0)
        
        return frequency_score
    
    def _compute_depth_score(self, gray):
        """
        Depth/focus analysis
        Real faces have depth variation, printed photos are flat
        Optimized for performance
        """
        # Use focus measure to estimate depth variation
        # Areas in focus vs out of focus indicate depth
        
        # Compute focus measure using variance of Laplacian in local regions
        h, w = gray.shape
        block_size = min(32, max(8, min(h, w) // 4))
        
        if block_size < 8:
            return 0.5  # Image too small for depth analysis
        
        # Use vectorized approach for better performance
        num_blocks_h = (h - block_size) // (block_size // 2) + 1
        num_blocks_w = (w - block_size) // (block_size // 2) + 1
        
        if num_blocks_h <= 0 or num_blocks_w <= 0:
            return 0.5
        
        focus_scores = []
        step = max(1, block_size // 2)
        
        for i in range(0, h - block_size + 1, step):
            for j in range(0, w - block_size + 1, step):
                block = gray[i:i+block_size, j:j+block_size]
                if block.size > 0:
                    laplacian = cv2.Laplacian(block, cv2.CV_64F)
                    focus_var = laplacian.var()
                    if not np.isnan(focus_var) and focus_var >= 0:
                        focus_scores.append(focus_var)
        
        if len(focus_scores) < 2:
            return 0.5
        
        # Real faces have varying focus (depth variation)
        # Printed photos are uniformly in focus (flat)
        focus_scores = np.array(focus_scores)
        focus_variance = np.var(focus_scores)
        focus_mean = np.mean(focus_scores)
        
        # Higher variance in focus = more depth variation = more likely real
        if focus_mean > 1e-10:
            depth_score = min(focus_variance / focus_mean * 0.5, 1.0)
        else:
            depth_score = 0.5
        
        return depth_score
    
    def _compute_motion_blur_score(self, gray):
        """
        Motion blur detection
        Real faces may have slight motion blur, printed photos are static
        """
        # Analyze edge sharpness in different directions
        # Motion blur creates directional blur
        
        # Compute gradients in x and y directions
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Analyze directional consistency
        # Motion blur creates asymmetric gradient patterns
        grad_x_abs = np.abs(grad_x)
        grad_y_abs = np.abs(grad_y)
        
        # Ratio of gradients indicates directional bias
        # Real faces: more balanced gradients
        # Printed photos: may have directional artifacts
        x_energy = np.mean(grad_x_abs)
        y_energy = np.mean(grad_y_abs)
        
        if x_energy + y_energy < 1e-10:
            return 0.5
        
        # Balance ratio (closer to 1.0 = more balanced = more likely real)
        balance_ratio = min(x_energy, y_energy) / max(x_energy, y_energy)
        
        # Slight motion blur (balance ratio 0.7-0.9) suggests real face
        # Perfect balance (1.0) or extreme imbalance suggests printed photo
        if 0.7 <= balance_ratio <= 0.95:
            motion_score = 0.8
        elif balance_ratio > 0.95:
            motion_score = 0.5  # Too perfect, might be printed
        else:
            motion_score = 0.4  # Too imbalanced
        
        return motion_score
    
    def detect_batch(self, images):
        """Detect liveness for batch of images"""
        results = []
        for img in images:
            is_real, confidence = self.detect(img)
            results.append((is_real, confidence))
        return results


def create_anti_spoofing_detector(method=None, device='cpu'):
    """Create anti-spoofing detector"""
    return AntiSpoofingDetector(method=method, device=device)

