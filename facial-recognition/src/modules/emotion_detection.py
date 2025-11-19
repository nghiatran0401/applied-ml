"""
Emotion Detection Module
Identifies emotional states from facial expressions
"""
import torch
import numpy as np
from PIL import Image
import cv2
import logging

logger = logging.getLogger(__name__)

# Lazy import - only import FER when actually needed
# This prevents TensorFlow/gRPC from loading at module import time
FER_AVAILABLE = None

def _check_fer_available():
    """Lazy check if FER is available"""
    global FER_AVAILABLE
    if FER_AVAILABLE is None:
        try:
            from fer import FER
            FER_AVAILABLE = True
        except ImportError:
            FER_AVAILABLE = False
    return FER_AVAILABLE

def _get_fer():
    """Lazy import of FER"""
    if _check_fer_available():
        from fer import FER
        return FER
    return None


class EmotionDetector:
    """
    Emotion detection using pre-trained models
    Detects: happy, sad, angry, surprise, fear, disgust, neutral
    """
    def __init__(self, method='fer', device='cpu'):
        """
        Args:
            method: 'fer' (FER library) or 'simple' (heuristic)
            device: 'cuda' or 'cpu'
        """
        self.method = method
        self.device = device
        
        if method == 'fer' and not _check_fer_available():
            logger.warning("fer library not available. Falling back to simple method.")
            self.method = 'simple'
        
        if self.method == 'fer':
            logger.info("Using FER library for emotion detection")
            # Lazy initialization - only create FER detector when needed
            self.detector = None  # Will be created on first use
        else:
            logger.info("Using simple emotion detection")
            self.detector = None
    
    def detect(self, image):
        """
        Detect emotion from face image
        
        Args:
            image: PIL Image or numpy array (face image)
            
        Returns:
            emotion: Detected emotion (string)
            confidence: Confidence score (0-1)
            all_emotions: Dictionary of all emotion scores
        """
        if self.method == 'fer':
            return self._detect_fer(image)
        else:
            return self._detect_simple(image)
    
    def _detect_fer(self, image):
        """
        Use FER library for emotion detection
        """
        try:
            # Lazy initialization of FER detector
            if self.detector is None:
                FER = _get_fer()
                if FER is None:
                    return self._detect_simple(image)
                self.detector = FER(mtcnn=True)  # Use MTCNN for face detection
            
            # Convert PIL to numpy if needed
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image
            
            # FER expects BGR format for OpenCV
            if len(img_array.shape) == 3:
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img_array
            
            # Detect emotions
            emotions = self.detector.detect_emotions(img_bgr)
            
            if len(emotions) == 0:
                # No face detected
                return 'neutral', 0.5, {'neutral': 1.0}
            
            # Get top emotion
            top_emotion = emotions[0]
            all_emotions = top_emotion['emotions']
            
            # Find emotion with highest score
            emotion = max(all_emotions, key=all_emotions.get)
            confidence = all_emotions[emotion]
            
            return emotion, float(confidence), all_emotions
            
        except Exception as e:
            logger.error(f"FER detection error: {e}")
            return self._detect_simple(image)
    
    def _detect_simple(self, image):
        """
        Simple heuristic-based emotion detection
        This is a placeholder - in production, use proper trained models
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
        
        # Simple heuristic: analyze facial features
        # This is very simplified - real emotion detection needs trained models
        
        # Compute some basic features
        # (In reality, you'd use facial landmarks and trained classifiers)
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        # Very simple heuristic based on image statistics
        # This is just a placeholder - not accurate
        if mean_intensity > 150:
            emotion = 'happy'
            confidence = 0.6
        elif mean_intensity < 100:
            emotion = 'sad'
            confidence = 0.6
        else:
            emotion = 'neutral'
            confidence = 0.7
        
        all_emotions = {
            'happy': 0.3,
            'sad': 0.2,
            'angry': 0.1,
            'surprise': 0.1,
            'fear': 0.1,
            'disgust': 0.1,
            'neutral': 0.1
        }
        all_emotions[emotion] = confidence
        
        return emotion, confidence, all_emotions
    
    def detect_batch(self, images):
        """
        Detect emotions for batch of images
        
        Args:
            images: List of PIL Images or numpy arrays
            
        Returns:
            results: List of (emotion, confidence, all_emotions) tuples
        """
        results = []
        for img in images:
            emotion, confidence, all_emotions = self.detect(img)
            results.append((emotion, confidence, all_emotions))
        return results
    
    def get_emotion_icon(self, emotion):
        """
        Get emoji icon for emotion
        
        Args:
            emotion: Emotion string
            
        Returns:
            icon: Emoji string
        """
        emotion_icons = {
            'happy': '😊',
            'sad': '😢',
            'angry': '😠',
            'surprise': '😲',
            'fear': '😨',
            'disgust': '🤢',
            'neutral': '😐'
        }
        return emotion_icons.get(emotion.lower(), '😐')


def create_emotion_detector(method='fer', device='cpu'):
    """
    Create emotion detector
    
    Args:
        method: 'fer' or 'simple'
        device: 'cuda', 'mps', or 'cpu'
        
    Returns:
        detector: EmotionDetector instance
    """
    return EmotionDetector(method=method, device=device)

