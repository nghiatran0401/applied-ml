"""
Face Database Management
Adapted from 7-face-recognition patterns but using our trained model embeddings
"""
import os
import json
import pickle
import numpy as np
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
import logging

from src.models.classification_model import create_model
from src.utils.face_detector import FaceDetector
from src.utils.config import get_config

logger = logging.getLogger(__name__)


class FaceDatabase:
    """
    Database for storing and retrieving face embeddings
    """
    def __init__(self, model_path, num_classes, device='cpu'):
        self.device = device
        self.model = create_model(num_classes=num_classes, pretrained=False)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model = self.model.to(device)
        
        # Face detector for alignment
        self.face_detector = FaceDetector()
        
        # Database: {person_id: [embeddings]}
        self.database = {}
        self.person_names = {}  # {person_id: name}
        self.avatar_paths = {}  # {person_id: avatar_image_path}
        
        # Transform for images
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_embedding(self, image_path):
        """Extract face embedding from image"""
        # Detect and align face
        face, bbox = self.face_detector.detect_and_align(image_path)
        if face is None:
            return None
        
        # Transform
        face_tensor = self.transform(face).unsqueeze(0).to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.model.extract_embedding(face_tensor)
            embedding = embedding.cpu().numpy().flatten()
            
            # Normalize embedding for better cosine similarity
            embedding_norm = np.linalg.norm(embedding)
            if embedding_norm > 0:
                embedding = embedding / embedding_norm
        
        return embedding
    
    def register_person(self, person_id, person_name, image_paths):
        """Register a person with multiple images"""
        embeddings = []
        for img_path in image_paths:
            embedding = self.extract_embedding(img_path)
            if embedding is not None:
                embeddings.append(embedding)
        
        if embeddings:
            # Normalize all embeddings before storing
            normalized_embeddings = []
            for emb in embeddings:
                emb_norm = np.linalg.norm(emb)
                if emb_norm > 0:
                    normalized_embeddings.append(emb / emb_norm)
                else:
                    normalized_embeddings.append(emb)
            
            self.database[person_id] = np.array(normalized_embeddings)
            self.person_names[person_id] = person_name
            # Store first image as avatar
            if image_paths:
                self.avatar_paths[person_id] = image_paths[0]
            logger.info(f"Registered {person_name} with {len(embeddings)} face embeddings")
        else:
            logger.warning(f"No valid faces found for {person_name}")
    
    def load_from_folder(self, folder_path):
        """Load known faces from folder structure"""
        folder = Path(folder_path)
        for person_folder in folder.iterdir():
            if person_folder.is_dir():
                person_name = person_folder.name
                person_id = len(self.database)
                
                # Get all images
                image_paths = list(person_folder.glob('*.jpg')) + \
                             list(person_folder.glob('*.png')) + \
                             list(person_folder.glob('*.jpeg'))
                
                if image_paths:
                    self.register_person(person_id, person_name, image_paths)
    
    def find_match(self, embedding, threshold=None, metric=None):
        """Find best matching person for given embedding"""
        config = get_config()
        if threshold is None:
            threshold = config.face_recognition.similarity_threshold_verify
        if metric is None:
            metric = config.face_recognition.metric
        
        if len(self.database) == 0:
            return None, None, 0.0
        
        best_match_id = None
        best_similarity = -1.0
        
        # Ensure input embedding is normalized
        embedding_norm = np.linalg.norm(embedding)
        if embedding_norm > 0:
            embedding = embedding / embedding_norm
        
        for person_id, embeddings in self.database.items():
            # Compute similarity with all embeddings for this person
            if metric == 'cosine':
                # Since embeddings are normalized, cosine similarity is just dot product
                similarities = np.dot(embeddings, embedding)
            else:  # euclidean
                distances = np.linalg.norm(embeddings - embedding, axis=1)
                similarities = 1 / (1 + distances)  # Convert distance to similarity
            
            max_sim = np.max(similarities)
            
            if max_sim > best_similarity:
                best_similarity = max_sim
                best_match_id = person_id
        
        if best_similarity >= threshold:
            person_name = self.person_names.get(best_match_id, f"Person_{best_match_id}")
            return best_match_id, person_name, best_similarity
        else:
            return None, None, best_similarity
    
    def save(self, filepath):
        """Save database to file"""
        data = {
            'database': {k: v.tolist() for k, v in self.database.items()},
            'person_names': self.person_names,
            'avatar_paths': self.avatar_paths
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Database saved to {filepath}")
    
    def load(self, filepath):
        """Load database from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.database = {k: np.array(v) for k, v in data['database'].items()}
        self.person_names = data['person_names']
        # Load avatar paths if they exist (for backward compatibility)
        self.avatar_paths = data.get('avatar_paths', {})
        logger.info(f"Database loaded from {filepath}: {len(self.database)} people")

