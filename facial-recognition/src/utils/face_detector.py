"""
Face Detection and Alignment using MTCNN
Enhanced with multi-face detection, quality assessment, and angle validation
"""
import cv2
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
import torch
import math

class FaceDetector:
    def __init__(self, device=None):
        if device is None:
            # Device selection: CUDA > CPU
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        """
        Initialize MTCNN face detector
        
        Args:
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=10,  # Lower minimum face size for better detection
            thresholds=[0.5, 0.6, 0.6],  # Lower thresholds for more lenient detection
            factor=0.709,
            post_process=False,
            device=device
        )
    
    def detect_and_align(self, image_path):
        """
        Detect face and return aligned face image
        
        Args:
            image_path: Path to image file
            
        Returns:
            aligned_face: PIL Image of aligned face (160x160)
            bounding_box: (x, y, width, height) or None if no face
        """
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')
            
            # Detect face and align
            aligned_face = self.mtcnn(img)
            
            if aligned_face is None:
                return None, None
        except Exception as e:
            return None, None
        
        # Convert tensor to PIL Image
        if isinstance(aligned_face, torch.Tensor):
            aligned_face = aligned_face.permute(1, 2, 0).cpu().numpy()
            aligned_face = (aligned_face * 255).astype(np.uint8)
            aligned_face = Image.fromarray(aligned_face)
        
        # Get bounding box (approximate, MTCNN doesn't return it directly)
        # For now, we'll use the aligned face
        bounding_box = (0, 0, 160, 160)  # Aligned faces are 160x160
        
        return aligned_face, bounding_box
    
    def detect_all_faces(self, image_path):
        """
        Detect all faces in image (for crowded environments)
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of dicts with keys: 'face' (PIL Image), 'bbox' (x1, y1, x2, y2), 
            'confidence', 'quality_score', 'angle_info'
        """
        # Load image
        img_cv = cv2.imread(image_path)
        if img_cv is None:
            return []
        
        img_pil = Image.open(image_path).convert('RGB')
        original_height, original_width = img_cv.shape[:2]
        
        # Use OpenCV Haar Cascade for multi-face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        
        detected_faces = []
        
        for (x, y, w, h) in faces:
            # Crop face region
            face_region = img_pil.crop((x, y, x + w, y + h))
            
            # Get aligned face using MTCNN (on the cropped region)
            aligned_face = self.mtcnn(face_region)
            
            if aligned_face is None:
                continue
            
            # Convert tensor to PIL Image
            if isinstance(aligned_face, torch.Tensor):
                aligned_face = aligned_face.permute(1, 2, 0).cpu().numpy()
                aligned_face = (aligned_face * 255).astype(np.uint8)
                aligned_face = Image.fromarray(aligned_face)
            
            # Calculate quality metrics
            quality_score = self._calculate_face_quality(aligned_face, w, h, original_width, original_height)
            angle_info = self._estimate_face_angle(w, h, x, y, original_width, original_height)
            
            detected_faces.append({
                'face': aligned_face,
                'bbox': (int(x), int(y), int(x + w), int(y + h)),
                'confidence': 0.9,  # Haar cascade doesn't provide confidence
                'quality_score': quality_score,
                'angle_info': angle_info
            })
        
        # Sort by quality score (best first)
        detected_faces.sort(key=lambda x: x['quality_score'], reverse=True)
        
        return detected_faces
    
    def _calculate_face_quality(self, face_image, face_width, face_height, img_width, img_height):
        """
        Calculate face quality score based on multiple factors
        
        Args:
            face_image: PIL Image of face
            face_width, face_height: Face dimensions in original image
            img_width, img_height: Original image dimensions
            
        Returns:
            Quality score (0.0 to 1.0, higher is better)
        """
        # Convert PIL to numpy for processing
        face_np = np.array(face_image.convert('L'))  # Grayscale
        
        # 1. Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(face_np, cv2.CV_64F)
        sharpness = laplacian.var()
        sharpness_score = min(sharpness / 500.0, 1.0)  # Normalize (500 is good threshold)
        
        # 2. Size (face should be reasonably large)
        face_area = face_width * face_height
        img_area = img_width * img_height
        size_ratio = face_area / img_area
        # Ideal: 5-15% of image
        if size_ratio < 0.02:  # Too small
            size_score = size_ratio / 0.02
        elif size_ratio > 0.30:  # Too large (might be cropped wrong)
            size_score = max(0.5, 1.0 - (size_ratio - 0.30) / 0.20)
        else:  # Good size
            size_score = 1.0
        
        # 3. Aspect ratio (face should be roughly square)
        aspect_ratio = face_width / face_height if face_height > 0 else 1.0
        # Ideal: 0.7 to 1.3
        if 0.7 <= aspect_ratio <= 1.3:
            aspect_score = 1.0
        else:
            aspect_score = max(0.0, 1.0 - abs(aspect_ratio - 1.0) / 0.5)
        
        # 4. Brightness (should be well-lit, not too dark or too bright)
        mean_brightness = np.mean(face_np)
        # Ideal: 80-200 (out of 255)
        if 80 <= mean_brightness <= 200:
            brightness_score = 1.0
        elif mean_brightness < 80:
            brightness_score = mean_brightness / 80.0
        else:  # > 200
            brightness_score = max(0.0, 1.0 - (mean_brightness - 200) / 55.0)
        
        # Weighted combination
        quality_score = (
            0.4 * sharpness_score +
            0.3 * size_score +
            0.15 * aspect_score +
            0.15 * brightness_score
        )
        
        return quality_score
    
    def _estimate_face_angle(self, face_width, face_height, face_x, face_y, img_width, img_height):
        """
        Estimate face pose angles (yaw, pitch, roll)
        Simplified estimation based on face position and size
        
        Args:
            face_width, face_height: Face dimensions
            face_x, face_y: Face position
            img_width, img_height: Image dimensions
            
        Returns:
            Dict with 'yaw', 'pitch', 'roll' estimates (in degrees)
        """
        # Yaw (left-right rotation): Estimate based on face position
        # If face is centered, yaw is ~0
        center_x = face_x + face_width / 2
        img_center_x = img_width / 2
        x_offset = (center_x - img_center_x) / img_width
        yaw = x_offset * 45  # Rough estimate, max ±45 degrees
        
        # Pitch (up-down rotation): Estimate based on face position
        center_y = face_y + face_height / 2
        img_center_y = img_height / 2
        y_offset = (center_y - img_center_y) / img_height
        pitch = y_offset * 45  # Rough estimate, max ±45 degrees
        
        # Roll (tilt): Hard to estimate without landmarks, assume 0 for now
        # In production, would use facial landmarks
        roll = 0.0
        
        return {
            'yaw': float(yaw),
            'pitch': float(pitch),
            'roll': float(roll),
            'is_valid': abs(yaw) <= 30 and abs(pitch) <= 30 and abs(roll) <= 15
        }
    
    def detect_and_align_with_quality(self, image_path, min_quality=0.5, require_valid_angle=True):
        """
        Detect face with quality and angle validation
        
        Args:
            image_path: Path to image file
            min_quality: Minimum quality score (0.0 to 1.0)
            require_valid_angle: If True, reject faces with extreme angles
            
        Returns:
            Tuple: (aligned_face, bbox, quality_info) or (None, None, None)
            quality_info: Dict with 'quality_score', 'angle_info', 'warnings'
        """
        # First, try to detect all faces
        all_faces = self.detect_all_faces(image_path)
        
        if len(all_faces) == 0:
            # Fallback to single face detection
            face, bbox = self.detect_and_align(image_path)
            if face is None:
                return None, None, None
            
            # Calculate quality for single face
            img_cv = cv2.imread(image_path)
            if img_cv is None:
                return None, None, None
            
            h, w = img_cv.shape[:2]
            quality_score = self._calculate_face_quality(face, 160, 160, w, h)
            angle_info = self._estimate_face_angle(160, 160, w//2-80, h//2-80, w, h)
            
            quality_info = {
                'quality_score': quality_score,
                'angle_info': angle_info,
                'warnings': []
            }
            
            # Check quality
            if quality_score < min_quality:
                quality_info['warnings'].append(f"Low quality image (score: {quality_score:.2f})")
            
            # Check angle
            if require_valid_angle and not angle_info['is_valid']:
                quality_info['warnings'].append(
                    f"Extreme face angle (yaw: {angle_info['yaw']:.1f}°, pitch: {angle_info['pitch']:.1f}°)"
                )
            
            return face, bbox, quality_info
        
        # Multiple faces detected - use the best quality one
        best_face = all_faces[0]
        
        # Check if multiple faces were found
        if len(all_faces) > 1:
            best_face['warnings'] = [f"Multiple faces detected ({len(all_faces)}). Using best quality face."]
        else:
            best_face['warnings'] = []
        
        # Check quality threshold
        if best_face['quality_score'] < min_quality:
            best_face['warnings'].append(
                f"Low quality image (score: {best_face['quality_score']:.2f}, minimum: {min_quality})"
            )
        
        # Check angle
        if require_valid_angle and not best_face['angle_info']['is_valid']:
            best_face['warnings'].append(
                f"Extreme face angle (yaw: {best_face['angle_info']['yaw']:.1f}°, "
                f"pitch: {best_face['angle_info']['pitch']:.1f}°). Please face camera directly."
            )
        
        quality_info = {
            'quality_score': best_face['quality_score'],
            'angle_info': best_face['angle_info'],
            'warnings': best_face['warnings'],
            'num_faces_detected': len(all_faces)
        }
        
        return best_face['face'], best_face['bbox'], quality_info

