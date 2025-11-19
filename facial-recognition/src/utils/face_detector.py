"""
Face Detection and Alignment using MTCNN
"""
import cv2
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
import torch

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
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
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
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Detect face and align
        aligned_face = self.mtcnn(img)
        
        if aligned_face is None:
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
    
    def detect_faces_batch(self, image_paths):
        """
        Detect faces in batch of images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of (aligned_face, bbox) tuples
        """
        results = []
        for img_path in image_paths:
            face, bbox = self.detect_and_align(img_path)
            results.append((face, bbox))
        return results

