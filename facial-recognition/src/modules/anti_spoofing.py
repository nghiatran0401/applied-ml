"""
Anti-Spoofing (Liveness Detection) Module
Detects fake faces (printed photos, screen photos)
"""
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import logging

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
    def __init__(self, method='deepface', device='cpu'):
        """
        Args:
            method: 'deepface' (pre-trained) or 'heuristic' (simple)
            device: 'cuda' or 'cpu'
        """
        self.method = method
        self.device = device
        
        if method == 'deepface' and not _check_deepface_available():
            logger.warning("deepface not available. Falling back to heuristic method.")
            self.method = 'heuristic'
        
        if method == 'deepface':
            logger.info("Using DeepFace for anti-spoofing detection")
        else:
            logger.info("Using heuristic-based anti-spoofing detection")
    
    def detect(self, image):
        """
        Detect if face is real or spoofed
        
        Args:
            image: PIL Image or numpy array (face image)
            
        Returns:
            is_real: True if real face, False if spoofed
            confidence: Confidence score (0-1, higher = more confident it's real)
        """
        if self.method == 'deepface':
            return self._detect_deepface(image)
        else:
            return self._detect_heuristic(image)
    
    def _detect_deepface(self, image):
        """
        Use DeepFace for liveness detection
        """
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
            # Note: DeepFace may not have direct liveness detection
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
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Higher variance = sharper image = more likely real
            if laplacian_var > 100:
                confidence = 0.9
            elif laplacian_var > 50:
                confidence = 0.7
            else:
                confidence = 0.5
            
            is_real = confidence > 0.6
            
            return is_real, confidence
            
        except Exception as e:
            logger.error(f"DeepFace detection error: {e}")
            return self._detect_heuristic(image)
    
    def _detect_heuristic(self, image):
        """
        Heuristic-based liveness detection
        Uses image quality, texture analysis, etc.
        """
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
        
        # Combine heuristics
        sharpness_score = min(laplacian_var / 200.0, 1.0)  # Normalize to 0-1
        texture_score_norm = min(texture_score / 50.0, 1.0)  # Normalize to 0-1
        
        # Weighted combination
        confidence = 0.4 * sharpness_score + 0.4 * texture_score_norm + 0.2 * color_score
        
        # Threshold
        is_real = confidence > 0.5
        
        return is_real, float(confidence)
    
    def _compute_texture_score(self, gray):
        """
        Compute texture variation score
        Higher = more texture = more likely real
        """
        # Compute gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Variance of gradient magnitude indicates texture
        texture_score = np.var(gradient_magnitude)
        
        return texture_score
    
    def _compute_color_score(self, img_array):
        """
        Compute color distribution score
        Real faces usually have more natural color variation
        """
        # Compute color variance in different channels
        color_var = np.var(img_array, axis=(0, 1))
        color_score = np.mean(color_var) / 100.0  # Normalize
        
        return min(color_score, 1.0)
    
    def detect_batch(self, images):
        """
        Detect liveness for batch of images
        
        Args:
            images: List of PIL Images or numpy arrays
            
        Returns:
            results: List of (is_real, confidence) tuples
        """
        results = []
        for img in images:
            is_real, confidence = self.detect(img)
            results.append((is_real, confidence))
        return results


def create_anti_spoofing_detector(method='heuristic', device='cpu'):
    """
    Create anti-spoofing detector
    
    Args:
        method: 'deepface' or 'heuristic'
        device: 'cuda', 'mps', or 'cpu'
        
    Returns:
        detector: AntiSpoofingDetector instance
    """
    return AntiSpoofingDetector(method=method, device=device)

