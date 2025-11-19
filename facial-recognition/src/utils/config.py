"""
Configuration Management System
Centralized configuration loading and validation using Pydantic
"""
import os
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
import logging

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Model configuration"""
    path: str = "models/classification_best.pth"
    num_classes: int = 4000
    embedding_dim: int = 2048
    metric_learning_embedding_dim: int = 512


class QualityConfig(BaseModel):
    """Face quality assessment configuration"""
    min_quality_verify: float = 0.5
    min_quality_realtime: float = 0.4
    min_quality_register: float = 0.0
    min_quality_check: float = 0.2
    sharpness_weight: float = 0.4
    size_weight: float = 0.3
    aspect_weight: float = 0.15
    brightness_weight: float = 0.15
    sharpness_threshold: float = 200.0
    min_size_ratio: float = 0.02
    max_size_ratio: float = 0.30
    ideal_size_ratio_min: float = 0.05
    ideal_size_ratio_max: float = 0.15


class AngleConfig(BaseModel):
    """Face angle validation configuration"""
    max_yaw: float = 30.0
    max_pitch: float = 30.0
    max_roll: float = 15.0
    require_valid_angle_verify: bool = True
    require_valid_angle_realtime: bool = False


class FaceDetectionConfig(BaseModel):
    """Face detection configuration"""
    mtcnn_thresholds: List[float] = [0.5, 0.6, 0.6]
    min_face_size: int = 10
    image_size: int = 160
    quality: QualityConfig = Field(default_factory=QualityConfig)
    angle: AngleConfig = Field(default_factory=AngleConfig)


class AntiSpoofingConfig(BaseModel):
    """Anti-spoofing configuration"""
    method: str = "enhanced_heuristic"
    confidence_threshold: float = 0.5
    
    # Enhanced heuristic weights
    sharpness_weight: float = 0.25
    texture_weight: float = 0.25
    color_weight: float = 0.15
    frequency_weight: float = 0.15
    depth_weight: float = 0.10
    motion_blur_weight: float = 0.10
    
    # Legacy heuristic weights (for backward compatibility)
    legacy_sharpness_weight: float = 0.4
    legacy_texture_weight: float = 0.4
    legacy_color_weight: float = 0.2
    
    sharpness_threshold_high: float = 200.0
    sharpness_threshold_medium: float = 100.0
    sharpness_threshold_low: float = 50.0
    confidence_high: float = 0.9
    confidence_medium: float = 0.7
    confidence_low: float = 0.5
    
    # Enhanced detection flags
    frequency_analysis_enabled: bool = True
    lbp_texture_enabled: bool = True
    depth_estimation_enabled: bool = True
    motion_blur_detection_enabled: bool = True


class EmotionDetectionConfig(BaseModel):
    """Emotion detection configuration"""
    method: str = "fer"
    default_emotion: str = "neutral"
    default_confidence: float = 0.5


class DatabaseConfig(BaseModel):
    """Database configuration"""
    face_database_path: str = "face_database.pkl"
    face_database_type: str = "pickle"
    attendance_logs_path: str = "attendance_logs.json"
    attendance_logs_type: str = "json"
    attendance_cooldown_minutes: int = 5


class FileUploadConfig(BaseModel):
    """File upload configuration"""
    max_file_size_mb: int = 10
    allowed_extensions: List[str] = [".jpg", ".jpeg", ".png"]
    temp_upload_dir: str = "temp_uploads"
    recognized_faces_dir: str = "recognized_faces"


class APIConfig(BaseModel):
    """API configuration"""
    rate_limit_per_minute: int = 60
    request_timeout_seconds: int = 30
    max_images_per_registration: int = 10


class RealtimeConfig(BaseModel):
    """Real-time processing configuration"""
    recognition_interval_ms: int = 200
    face_tracking_threshold_px: int = 50
    frame_resize_factor: float = 0.25
    min_quality: float = 0.4
    require_valid_angle: bool = False


class FaceRecognitionConfig(BaseModel):
    """Face recognition configuration"""
    similarity_threshold_verify: float = 0.85
    similarity_threshold_realtime: float = 0.6
    similarity_threshold_duplicate: float = 0.9
    metric: str = "cosine"
    model: ModelConfig = Field(default_factory=ModelConfig)


class AppConfig(BaseModel):
    """Main application configuration"""
    face_recognition: FaceRecognitionConfig = Field(default_factory=FaceRecognitionConfig)
    face_detection: FaceDetectionConfig = Field(default_factory=FaceDetectionConfig)
    anti_spoofing: AntiSpoofingConfig = Field(default_factory=AntiSpoofingConfig)
    emotion_detection: EmotionDetectionConfig = Field(default_factory=EmotionDetectionConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    file_upload: FileUploadConfig = Field(default_factory=FileUploadConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    realtime: RealtimeConfig = Field(default_factory=RealtimeConfig)
    logging: Dict[str, Any] = Field(default_factory=lambda: {"level": "INFO"})
    device: Dict[str, Any] = Field(default_factory=lambda: {"preferred": "auto"})
    
    @validator('face_recognition', 'face_detection', 'anti_spoofing', 
              'emotion_detection', 'database', 'file_upload', 'api', 'realtime',
              pre=True, always=True)
    def set_defaults(cls, v):
        """Set defaults if section is missing"""
        return v or {}


# Global config instance
_config: Optional[AppConfig] = None


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load configuration from YAML file
    """
    global _config
    
    if _config is not None:
        return _config
    
    if config_path is None:
        # Look for config.yaml in project root
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Config file not found at {config_path}. Using defaults.")
        _config = AppConfig()
        return _config
    
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create config with defaults for missing sections
        _config = AppConfig(**config_dict)
        logger.info(f"Configuration loaded from {config_path}")
        return _config
    
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        logger.warning("Using default configuration")
        _config = AppConfig()
        return _config


def get_config() -> AppConfig:
    """
    Get the current configuration instance
    """
    if _config is None:
        return load_config()
    return _config


def reload_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Reload configuration from file (useful for testing or hot-reloading)
    """
    global _config
    _config = None
    return load_config(config_path)


def save_config(config: AppConfig, config_path: Optional[str] = None):
    """
    Save configuration to YAML file
    """
    if config_path is None:
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config.yaml"
    
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict and save as YAML
    config_dict = config.dict()
    
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Configuration saved to {config_path}")


# Convenience functions for common config access
def get_similarity_threshold(mode: str = "verify") -> float:
    """Get similarity threshold for given mode"""
    config = get_config()
    if mode == "verify":
        return config.face_recognition.similarity_threshold_verify
    elif mode == "realtime":
        return config.face_recognition.similarity_threshold_realtime
    elif mode == "duplicate":
        return config.face_recognition.similarity_threshold_duplicate
    else:
        return config.face_recognition.similarity_threshold_verify


def get_model_path() -> str:
    """Get model path from config"""
    return get_config().face_recognition.model.path


def get_database_path() -> str:
    """Get face database path from config"""
    return get_config().database.face_database_path


def get_attendance_logs_path() -> str:
    """Get attendance logs path from config"""
    return get_config().database.attendance_logs_path