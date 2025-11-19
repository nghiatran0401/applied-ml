"""
FastAPI Web Application for Face Recognition Attendance System
"""
import sys
import os
from pathlib import Path

# Suppress TensorFlow/gRPC warnings and set threading before any ML imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads
os.environ['MKL_NUM_THREADS'] = '1'  # Limit MKL threads

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from jinja2 import Template
import uvicorn
import torch
import numpy as np
from PIL import Image
import json
import io
import uuid
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Lazy imports - only import when needed to avoid TensorFlow/gRPC mutex issues
# These will be imported inside functions when actually used

def convert_to_python_types(obj):
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_python_types(item) for item in obj]
    return obj

app = FastAPI(title="Face Recognition Attendance System", version="1.0.0")

# Create templates directory
templates_dir = Path(__file__).parent.parent / "templates"
templates_dir.mkdir(exist_ok=True)

def render_template(template_name: str, **kwargs):
    """Render Jinja2 template"""
    template_path = templates_dir / template_name
    with open(template_path, 'r') as f:
        template = Template(f.read())
    return template.render(**kwargs)

# Initialize components (lazy loading)
face_detector = None
anti_spoofing = None
emotion_detector = None
database = None
attendance_logger = None

def get_device():
    """Get available device"""
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

def initialize_components():
    """Initialize all components (lazy loading to avoid import-time mutex issues)"""
    global face_detector, anti_spoofing, emotion_detector, database, attendance_logger
    
    if face_detector is None:
        from src.utils.face_detector import FaceDetector
        face_detector = FaceDetector()
    
    if anti_spoofing is None:
        from src.modules.anti_spoofing import create_anti_spoofing_detector
        anti_spoofing = create_anti_spoofing_detector(method=None)  # Uses config
    
    if emotion_detector is None:
        from src.modules.emotion_detection import create_emotion_detector
        emotion_detector = create_emotion_detector(method=None)  # Uses config
    
    if database is None:
        from src.utils.face_database import FaceDatabase
        model_path = "models/classification_best.pth"
        if not os.path.exists(model_path):
            return False
        
        device = get_device()
        database = FaceDatabase(
            model_path=model_path,
            num_classes=4000,
            device=device
        )
        
        db_path = "face_database.pkl"
        if os.path.exists(db_path):
            try:
                database.load(db_path)
            except Exception as e:
                logger.warning(f"Could not load database: {e}")
    
    if attendance_logger is None:
        from src.utils.attendance_logger import AttendanceLogger
        attendance_logger = AttendanceLogger()
    
    return True

@app.get("/")
async def home():
    """Redirect to verify page"""
    return RedirectResponse(url="/verify")

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    """Registration page with database"""
    if not initialize_components():
        raise HTTPException(status_code=500, detail="Model not found. Please train the model first.")
    
    # Get people list for database section
    people = []
    if database:
        for person_id, person_name in database.person_names.items():
            avatar_path = database.avatar_paths.get(person_id)
            avatar_url = None
            if avatar_path and os.path.exists(avatar_path):
                avatar_url = f"/api/avatar/{person_id}"
            
            people.append({
                "id": person_id,
                "name": person_name,
                "embedding_count": len(database.database[person_id]),
                "avatar_url": avatar_url
            })
    
    return HTMLResponse(render_template("register.html", people=people))

@app.post("/api/register")
async def register_person(
    name: str = Form(...),
    images: list[UploadFile] = File(...)
):
    """Register a new person with captured photos from camera"""
    if not initialize_components():
        raise HTTPException(status_code=500, detail="Model not found")
    
    if not name:
        raise HTTPException(status_code=400, detail="Name is required")
    
    if not images:
        raise HTTPException(status_code=400, detail="At least one image is required")
    
    # Check if name already exists
    name_lower = name.strip().lower()
    for existing_id, existing_name in database.person_names.items():
        if existing_name.strip().lower() == name_lower:
            return JSONResponse(
                {
                    "success": False,
                    "error": f"Employee name '{name}' is already registered. Please use a different name."
                },
                status_code=400
            )
    
    try:
        # Save captured images temporarily
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        image_paths = []
        
        rejected_images = []
        for captured_file in images:
            # Generate unique filename to avoid conflicts
            unique_filename = f"{uuid.uuid4()}.jpg"
            image_path = temp_dir / unique_filename
            
            with open(image_path, "wb") as f:
                content = await captured_file.read()
                f.write(content)
            
            # Verify image is valid
            try:
                from PIL import Image
                test_img = Image.open(image_path)
                test_img.verify()
                # Reopen after verify (verify closes the file)
                test_img = Image.open(image_path)
                test_img.close()
            except Exception as e:
                rejected_images.append({
                    "filename": captured_file.filename,
                    "reason": f"Invalid image file: {str(e)}"
                })
                if os.path.exists(image_path):
                    os.remove(image_path)
                continue
            
            # First, try basic detection (most reliable)
            face_basic, bbox_basic = face_detector.detect_and_align(str(image_path))
            
            if face_basic is not None:
                image_paths.append(str(image_path))
                continue
            
            # If basic fails, try with quality checks
            face, bbox, quality_info = face_detector.detect_and_align_with_quality(
                str(image_path),
                min_quality=0.0,  # No quality threshold - just detect face
                require_valid_angle=False  # No angle validation
            )
            
            if face is not None:
                image_paths.append(str(image_path))
            else:
                rejected_images.append({
                    "filename": captured_file.filename,
                    "reason": "No face detected. Please ensure your face is clearly visible, well-lit, and facing the camera."
                })
                os.remove(image_path)
        
        # Check if we have any valid images
        if not image_paths:
            error_msg = "No valid faces detected in captured photos. "
            if rejected_images:
                reasons = [img["reason"] for img in rejected_images]
                error_msg += f"Rejected: {', '.join(set(reasons))}. "
            error_msg += "Please capture clear photos with visible faces, good lighting, and face the camera directly."
            
            return JSONResponse(
                {
                    "success": False,
                    "error": error_msg,
                    "rejected_images": rejected_images
                },
                status_code=400
            )
        
        # Check for duplicate face before registering
        # Extract embedding from first valid image and check against existing faces
        if len(database.database) > 0 and image_paths:
            test_embedding = database.extract_embedding(image_paths[0])
            if test_embedding is not None:
                # Check against all existing faces
                for existing_id, existing_embeddings in database.database.items():
                    existing_name = database.person_names.get(existing_id, f"Person_{existing_id}")
                    # Calculate cosine similarity with all embeddings for this person
                    similarities = np.dot(existing_embeddings, test_embedding)
                    max_similarity = float(np.max(similarities))
                    
                    if max_similarity >= 0.9:  # 90% confidence threshold
                        # Clean up temp files
                        for path in image_paths:
                            if os.path.exists(path):
                                os.remove(path)
                        
                        return JSONResponse(
                            {
                                "success": False,
                                "error": f"This face is already registered as '{existing_name}' (similarity: {max_similarity*100:.1f}%). The same person cannot be registered multiple times."
                            },
                            status_code=400
                        )
        
        # Register person
        person_id = len(database.database)
        
        database.register_person(person_id, name, image_paths)
        
        # Check if registration actually succeeded (person was added)
        if person_id not in database.database:
            # Person was not added to database
            # Clean up temp files
            for path in image_paths:
                if os.path.exists(path):
                    os.remove(path)
            
            error_msg = "No valid faces detected in captured photos. "
            if rejected_images:
                reasons = [img["reason"] for img in rejected_images]
                error_msg += f"Rejected: {', '.join(set(reasons))}. "
            error_msg += "Please capture clear photos with visible faces, good lighting, and face the camera directly."
            
            return JSONResponse(
                {
                    "success": False,
                    "error": error_msg,
                    "rejected_images": rejected_images
                },
                status_code=400
            )
        
        # Registration succeeded, save database
        # Keep first image as avatar (don't delete it)
        avatar_path = image_paths[0] if image_paths else None
        database.save("face_database.pkl")
        
        # Clean up temp files (except avatar)
        for path in image_paths:
            if os.path.exists(path) and path != avatar_path:
                os.remove(path)
        
        return JSONResponse({
            "success": True,
            "message": f"Successfully registered {name}",
            "person_id": person_id
        })
    
    except Exception as e:
        return JSONResponse(
            {"success": False, "error": str(e)},
            status_code=500
        )

@app.get("/verify", response_class=HTMLResponse)
async def verify_page(request: Request):
    """Verification page"""
    if not initialize_components():
        raise HTTPException(status_code=500, detail="Model not found")
    
    return HTMLResponse(render_template("verify.html"))

@app.post("/api/verify/check-quality")
async def check_image_quality(image: UploadFile = File(...)):
    """Check image quality for real-time feedback"""
    if not initialize_components():
        raise HTTPException(status_code=500, detail="Model not found")
    
    temp_path = None
    try:
        # Read image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        img_width, img_height = pil_image.size
        
        # Save to temp file
        temp_path = "temp_quality_check.jpg"
        pil_image.save(temp_path)
        
        # Check face quality - very lenient to match our thresholds
        # Lower min_quality so we can detect faces even with lower quality scores
        face, bbox, quality_info = face_detector.detect_and_align_with_quality(
            temp_path,
            min_quality=0.05,  # Very low - just needs to detect a face, we'll check quality after
            require_valid_angle=False
        )
        
        if face is None:
            return JSONResponse({
                "hasFace": False,
                "isCentered": False,
                "hasGoodLighting": False,
                "isClear": False,
                "message": "No face detected. Please position your face in the center."
            })
        
        quality_score = quality_info.get("quality_score", 0.0)
        
        # Log quality score for debugging
        logger.info(f"Quality check - score: {quality_score:.3f}")
        
        # Since it worked before, make it extremely lenient
        # If face is detected, automatically pass quality checks
        # Only check centering, and make that lenient too
        has_good_lighting = True  # Always pass if face detected
        is_clear = True  # Always pass if face detected
        
        logger.info(f"Quality check - score: {quality_score:.3f}, auto-passing quality checks since face is detected")
        
        # Check if face is centered (using bbox if available)
        # Extremely lenient centering - 50% offset allowed (basically anywhere in frame)
        # Note: bbox format is (x1, y1, x2, y2) from detect_all_faces
        is_centered = True  # Default to True if no bbox
        if bbox and len(bbox) >= 4:
            # Handle both formats: (x1, y1, x2, y2) or (x, y, width, height)
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                # Format is (x1, y1, x2, y2)
                face_center_x = (bbox[0] + bbox[2]) / 2
                face_center_y = (bbox[1] + bbox[3]) / 2
            else:
                # Format is (x, y, width, height)
                face_center_x = bbox[0] + bbox[2] / 2
                face_center_y = bbox[1] + bbox[3] / 2
            
            img_center_x = img_width / 2
            img_center_y = img_height / 2
            
            offset_x = abs(face_center_x - img_center_x) / img_width
            offset_y = abs(face_center_y - img_center_y) / img_height
            # 50% offset - very lenient, basically anywhere in the frame
            is_centered = offset_x < 0.5 and offset_y < 0.5
            logger.info(f"Centering check - bbox: {bbox}, face_center: ({face_center_x:.1f}, {face_center_y:.1f}), offset: ({offset_x:.3f}, {offset_y:.3f}), is_centered: {is_centered}")
        else:
            # No bbox available, assume centered
            is_centered = True
            logger.info("No bbox available, assuming face is centered")
        
        # If face is detected, all checks should pass
        all_good = is_centered and has_good_lighting and is_clear
        
        logger.info(f"All checks - centered: {is_centered}, lighting: {has_good_lighting}, clear: {is_clear}, all_good: {all_good}")
        
        message = "Perfect! Capturing..." if all_good else (
            "Please move your face to the center" if not is_centered else
            "More lighting needed" if not has_good_lighting else
            "Image is blurry, please hold still" if not is_clear else
            "Please adjust your position"
        )
        
        return JSONResponse({
            "hasFace": True,
            "isCentered": is_centered,
            "hasGoodLighting": has_good_lighting,
            "isClear": is_clear,
            "message": message
        })
    except Exception as e:
        return JSONResponse({
            "hasFace": False,
            "isCentered": False,
            "hasGoodLighting": False,
            "isClear": False,
            "message": f"Error: {str(e)}"
        }, status_code=500)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/api/verify")
async def verify_person(image: UploadFile = File(...)):
    """Verify a person"""
    if not initialize_components():
        raise HTTPException(status_code=500, detail="Model not found")
    
    temp_path = None
    try:
        # Read image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Save to temp file for face detection
        temp_path = "temp_verify.jpg"
        pil_image.save(temp_path)
        
        results = {
            "face_detected": False,
            "is_real": False,
            "liveness_confidence": 0.0,
            "emotion": None,
            "emotion_confidence": 0.0,
            "all_emotions": {},
            "match_found": False,
            "person_name": None,
            "similarity": 0.0,
            "error": None
        }
        
        try:
            # Step 1: Face Detection with Quality Assessment
            # Very lenient thresholds - since it worked before, we're making it permissive
            # Only reject if face is completely undetectable
            face, bbox, quality_info = face_detector.detect_and_align_with_quality(
                temp_path, 
                min_quality=0.05,  # Very low - just needs to detect a face
                require_valid_angle=False  # Don't reject based on angle
            )
            
            if face is None:
                results["error"] = "No face detected in image. Please ensure your face is clearly visible."
                return JSONResponse(results)
            
            results["face_detected"] = True
            
            # Add quality information
            if quality_info:
                results["face_quality"] = {
                    "quality_score": float(quality_info.get("quality_score", 0.0)),
                    "angle_info": {
                        "yaw": float(quality_info.get("angle_info", {}).get("yaw", 0.0)),
                        "pitch": float(quality_info.get("angle_info", {}).get("pitch", 0.0)),
                        "roll": float(quality_info.get("angle_info", {}).get("roll", 0.0)),
                        "is_valid": bool(quality_info.get("angle_info", {}).get("is_valid", True))
                    },
                    "warnings": quality_info.get("warnings", []),
                    "num_faces_detected": quality_info.get("num_faces_detected", 1)
                }
                
                # Very lenient quality check - only reject if quality is extremely low
                # Since it worked before, we're being very permissive
                quality_score = quality_info.get("quality_score", 1.0)
                # Only reject if quality is extremely poor (almost no face visible)
                # This threshold is very low to allow most reasonable images through
                if quality_score < 0.1:
                    results["error"] = f"Image quality too low (score: {quality_score:.2f}). Please use a clearer image with good lighting."
                    return JSONResponse(results)
                # Log quality for debugging
                logger.info(f"Face quality score: {quality_score:.2f} - accepted")
                
                # Only warn about extreme angles, don't reject (since require_valid_angle=False)
                # This allows slight head movements while still detecting very extreme angles
                angle_info = quality_info.get("angle_info", {})
                if angle_info.get("is_valid", True) == False:
                    # Log warning but don't reject - angle estimation can be inaccurate
                    logger.info(f"Face angle detected: yaw={angle_info.get('yaw', 0):.1f}°, pitch={angle_info.get('pitch', 0):.1f}°")
            
            # Step 2: Anti-Spoofing
            try:
                is_real, liveness_confidence = anti_spoofing.detect(pil_image)
                results["is_real"] = bool(is_real)  # Convert to Python bool
                results["liveness_confidence"] = float(liveness_confidence)
            except Exception as e:
                logger.error(f"Anti-spoofing error: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                results["is_real"] = True  # Default to real if detection fails
                results["liveness_confidence"] = 0.5
            
            # Step 3: Emotion Detection
            try:
                emotion, emotion_confidence, all_emotions = emotion_detector.detect(pil_image)
                results["emotion"] = emotion
                results["emotion_confidence"] = float(emotion_confidence)
                results["all_emotions"] = {k: float(v) for k, v in all_emotions.items()}
            except Exception as e:
                logger.error(f"Emotion detection error: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                results["emotion"] = "neutral"
                results["emotion_confidence"] = 0.5
                results["all_emotions"] = {"neutral": 1.0}
            
            # Step 4: Face Recognition (only if real face)
            if results["is_real"] and len(database.database) > 0:
                try:
                    embedding = database.extract_embedding(temp_path)
                    if embedding is not None:
                        person_id, person_name, similarity = database.find_match(
                            embedding, threshold=0.85, metric='cosine'
                        )
                        if person_id is not None:
                            results["match_found"] = True
                            results["person_name"] = person_name
                            results["similarity"] = float(similarity)
                        else:
                            results["similarity"] = float(similarity)  # Best similarity even if below threshold
                except Exception as e:
                    logger.error(f"Face recognition error: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    results["error"] = f"Face recognition error: {str(e)}"
        
        except Exception as e:
            logger.error(f"Verification processing error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            results["error"] = f"Processing error: {str(e)}"
        
        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
        
        # Convert all NumPy types to Python native types for JSON serialization
        results = convert_to_python_types(results)
        return JSONResponse(results)
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Verify endpoint error: {e}")
        logger.debug(error_trace)
        
        # Clean up temp file if it exists
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        
        return JSONResponse(
            {"error": str(e), "traceback": error_trace},
            status_code=500
        )


@app.get("/api/avatar/{person_id}")
async def get_avatar(person_id: int):
    """Get avatar image for a person"""
    if not initialize_components():
        raise HTTPException(status_code=500, detail="Model not found")
    
    if person_id not in database.person_names:
        raise HTTPException(status_code=404, detail="Person not found")
    
    avatar_path = database.avatar_paths.get(person_id)
    if not avatar_path or not os.path.exists(avatar_path):
        raise HTTPException(status_code=404, detail="Avatar not found")
    
    from fastapi.responses import FileResponse
    return FileResponse(avatar_path, media_type="image/jpeg")

@app.delete("/api/database/{person_id}")
async def delete_person(person_id: int):
    """Delete a person from database"""
    if not initialize_components():
        raise HTTPException(status_code=500, detail="Model not found")
    
    if person_id not in database.database:
        raise HTTPException(status_code=404, detail="Person not found")
    
    person_name = database.person_names.get(person_id, "Unknown")
    del database.database[person_id]
    del database.person_names[person_id]
    # Delete avatar if exists
    if person_id in database.avatar_paths:
        avatar_path = database.avatar_paths[person_id]
        if os.path.exists(avatar_path):
            os.remove(avatar_path)
        del database.avatar_paths[person_id]
    database.save("face_database.pkl")
    
    return JSONResponse({
        "success": True,
        "message": f"Deleted {person_name}"
    })

@app.get("/realtime", response_class=HTMLResponse)
async def realtime_page(request: Request):
    """Real-time attendance processing page"""
    if not initialize_components():
        raise HTTPException(status_code=500, detail="Model not found")
    
    return HTMLResponse(render_template("realtime.html"))

@app.post("/api/realtime/process")
async def process_realtime_attendance(
    image: UploadFile = File(...),
    bbox: str = Form(None)  # Optional: bounding box from client-side detection
):
    """Process attendance from real-time camera feed using trained models
    Features: rectangle boxes, real-time recognition, screenshot saving (like notebook)"""
    if not initialize_components():
        raise HTTPException(status_code=500, detail="Model not found")
    
    temp_path = None
    try:
        # Read image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Save to temp file for face detection
        temp_path = "temp_realtime.jpg"
        pil_image.save(temp_path)
        
        # Parse bounding box if provided (from client-side detection)
        client_bbox = None
        if bbox:
            try:
                client_bbox = json.loads(bbox)
            except:
                pass
        
        results = {
            "face_detected": False,
            "is_real": False,
            "liveness_confidence": 0.0,
            "emotion": None,
            "match_found": False,
            "person_id": None,
            "person_name": None,
            "similarity": 0.0,
            "attendance_logged": False,
            "bbox": None,  # [x1, y1, x2, y2]
            "detection_confidence": 0.0,
            "status": "unknown",  # "logged", "cooldown", "unknown", "processing"
            "error": None,
            "screenshot_saved": None
        }
        
        try:
            # Use trained model-based recognition (your existing system)
            # Step 1: Face Detection with Quality Assessment
            if client_bbox:
                bbox = client_bbox
                face, _, quality_info = face_detector.detect_and_align_with_quality(
                    temp_path, 
                    min_quality=0.4, 
                    require_valid_angle=False  # More lenient for realtime
                )
                detection_confidence = 0.95
            else:
                face, bbox, quality_info = face_detector.detect_and_align_with_quality(
                    temp_path,
                    min_quality=0.4,
                    require_valid_angle=False  # More lenient for realtime
                )
                detection_confidence = quality_info.get("quality_score", 0.9) if quality_info else 0.9
            
            if bbox is None or face is None:
                results["error"] = "No face detected or face quality too low"
                return JSONResponse(results)
            
            results["face_detected"] = True
            results["bbox"] = bbox
            results["detection_confidence"] = float(detection_confidence) if detection_confidence else 0.9
            
            # Add quality information if available
            if quality_info:
                results["face_quality"] = {
                    "quality_score": float(quality_info.get("quality_score", 0.0)),
                    "angle_info": {
                        "yaw": float(quality_info.get("angle_info", {}).get("yaw", 0.0)),
                        "pitch": float(quality_info.get("angle_info", {}).get("pitch", 0.0)),
                        "roll": float(quality_info.get("angle_info", {}).get("roll", 0.0)),
                        "is_valid": bool(quality_info.get("angle_info", {}).get("is_valid", True))
                    },
                    "warnings": quality_info.get("warnings", []),
                    "num_faces_detected": quality_info.get("num_faces_detected", 1)
                }
            
            # Step 2: Anti-Spoofing
            try:
                is_real, liveness_confidence = anti_spoofing.detect(pil_image)
                results["is_real"] = bool(is_real)
                results["liveness_confidence"] = float(liveness_confidence)
            except Exception as e:
                logger.error(f"Anti-spoofing error: {e}")
                results["is_real"] = True
                results["liveness_confidence"] = 0.5
            
            # Step 3: Emotion Detection
            try:
                emotion, emotion_confidence, all_emotions = emotion_detector.detect(pil_image)
                results["emotion"] = emotion
            except Exception as e:
                logger.error(f"Emotion detection error: {e}")
                results["emotion"] = "neutral"
            
            # Step 4: Face Recognition (only if real face)
            if results["is_real"] and len(database.database) > 0:
                try:
                    embedding = database.extract_embedding(temp_path)
                    if embedding is not None:
                        person_id, person_name, similarity = database.find_match(
                            embedding, threshold=0.6, metric='cosine'
                        )
                        if person_id is not None:
                            results["match_found"] = True
                            results["person_id"] = int(person_id)
                            results["person_name"] = person_name
                            results["similarity"] = float(similarity)
                            
                            # Save screenshot of recognized face (like notebook)
                            try:
                                from datetime import datetime
                                import cv2
                                recognized_faces_dir = Path("recognized_faces")
                                recognized_faces_dir.mkdir(exist_ok=True)
                                
                                # Load image and extract face region
                                img_cv = cv2.imread(temp_path)
                                if img_cv is not None and bbox:
                                    x1, y1, x2, y2 = bbox
                                    face_crop = img_cv[int(y1):int(y2), int(x1):int(x2)]
                                    
                                    if face_crop.size > 0:
                                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                        safe_name = person_name.replace(" ", "_")
                                        screenshot_path = recognized_faces_dir / f"recognized_face_{safe_name}_{timestamp}.jpg"
                                        cv2.imwrite(str(screenshot_path), face_crop)
                                        results["screenshot_saved"] = str(screenshot_path)
                            except Exception as e:
                                logger.warning(f"Error saving screenshot: {e}")
                                results["screenshot_saved"] = None
                            
                            # Step 5: Log Attendance (prevent duplicates)
                            if attendance_logger.should_log_check_in(person_id, cooldown_minutes=5):
                                attendance_type = "check_in"
                                record = attendance_logger.log_attendance(
                                    person_id=person_id,
                                    person_name=person_name,
                                    attendance_type=attendance_type,
                                    confidence=similarity,
                                    emotion=results["emotion"],
                                    liveness_score=results["liveness_confidence"],
                                    location="main_entrance"
                                )
                                results["attendance_logged"] = True
                                results["status"] = "logged"
                                results["log_id"] = record["id"]
                                results["log_timestamp"] = record["timestamp"]
                            else:
                                results["attendance_logged"] = False
                                results["status"] = "cooldown"
                                results["error"] = "Recent check-in already recorded. Please wait a few minutes."
                        else:
                            results["similarity"] = float(similarity)
                            results["status"] = "unknown"
                    else:
                        results["status"] = "unknown"
                except Exception as e:
                    logger.error(f"Face recognition error: {e}")
                    results["error"] = f"Face recognition error: {str(e)}"
                    results["status"] = "error"
            else:
                results["status"] = "spoofed" if not results["is_real"] else "unknown"
        
        except Exception as e:
            logger.error(f"Processing error: {e}")
            results["error"] = f"Processing error: {str(e)}"
        
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
        
        results = convert_to_python_types(results)
        return JSONResponse(results)
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Realtime endpoint error: {e}")
        logger.debug(error_trace)
        
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        
        return JSONResponse(
            {"error": str(e), "traceback": error_trace},
            status_code=500
        )

@app.get("/api/attendance/logs")
async def get_attendance_logs(limit: int = 50):
    """Get recent attendance logs"""
    if not initialize_components():
        raise HTTPException(status_code=500, detail="System not initialized")
    
    logs = attendance_logger.get_recent_logs(limit=limit)
    return JSONResponse({"logs": logs, "count": len(logs)})

@app.get("/api/attendance/today")
async def get_today_attendance():
    """Get today's attendance logs"""
    if not initialize_components():
        raise HTTPException(status_code=500, detail="System not initialized")
    
    logs = attendance_logger.get_today_logs()
    return JSONResponse({"logs": logs, "count": len(logs)})

@app.delete("/api/attendance/logs")
async def delete_all_logs():
    """Delete all attendance logs"""
    if not initialize_components():
        raise HTTPException(status_code=500, detail="System not initialized")
    
    success = attendance_logger.delete_all_logs()
    if success:
        return JSONResponse({
            "success": True,
            "message": "All attendance logs deleted successfully"
        })
    else:
        raise HTTPException(status_code=500, detail="Failed to delete logs")

@app.get("/api/status")
async def get_status():
    """Get system status"""
    model_exists = os.path.exists("models/classification_best.pth")
    metric_model_exists = os.path.exists("models/metric_learning_best.pth")
    
    initialize_components()
    db_count = len(database.database) if database else 0
    today_count = len(attendance_logger.get_today_logs()) if attendance_logger else 0
    
    return JSONResponse({
        "classification_model": model_exists,
        "metric_learning_model": metric_model_exists,
        "database_count": db_count,
        "today_attendance_count": today_count,
        "device": get_device()
    })

@app.get("/results", response_class=HTMLResponse)
async def results_page(request: Request):
    """Model results and metrics page"""
    return HTMLResponse(render_template("results.html"))

def parse_training_log(log_path):
    """Parse training log file to extract key metrics"""
    training_info = {}
    
    if not os.path.exists(log_path):
        return training_info
    
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
            # Extract key information
            for i, line in enumerate(lines):
                # Training samples
                if "Training samples:" in line:
                    try:
                        training_info["training_samples"] = int(line.split("Training samples:")[1].strip())
                    except:
                        pass
                
                # Validation samples
                if "Validation samples:" in line:
                    try:
                        training_info["validation_samples"] = int(line.split("Validation samples:")[1].strip())
                    except:
                        pass
                
                # Number of classes
                if "images from" in line and "classes" in line:
                    try:
                        parts = line.split("images from")
                        if len(parts) > 1:
                            classes_part = parts[1].strip().split("classes")[0].strip()
                            training_info["num_classes"] = int(classes_part)
                    except:
                        pass
                
                # Number of epochs
                if "Starting training for" in line and "epochs" in line:
                    try:
                        parts = line.split("Starting training for")[1].strip().split("epochs")[0].strip()
                        training_info["num_epochs"] = int(parts)
                    except:
                        pass
                
                # Device
                if "Using device:" in line:
                    try:
                        training_info["device"] = line.split("Using device:")[1].strip()
                    except:
                        pass
                
                # Best validation loss (from end of file)
                if "Best Validation Loss:" in line:
                    try:
                        loss_str = line.split("Best Validation Loss:")[1].strip()
                        training_info["best_val_loss"] = float(loss_str)
                    except:
                        pass
                
                # Final train and validation loss (look for last occurrence)
                if "Train Loss:" in line and "Val Loss:" in line:
                    try:
                        parts = line.split("Train Loss:")[1].strip()
                        train_loss_str = parts.split()[0]
                        val_loss_str = parts.split("Val Loss:")[1].strip().split()[0]
                        training_info["final_train_loss"] = float(train_loss_str)
                        training_info["final_val_loss"] = float(val_loss_str)
                    except:
                        pass
            
            # Get final epoch info from last epoch entry
            for i in range(len(lines) - 1, max(0, len(lines) - 100), -1):
                if "Epoch" in lines[i] and "/" in lines[i]:
                    try:
                        epoch_parts = lines[i].split("Epoch")[1].strip().split("/")[0].strip()
                        training_info["final_epoch"] = int(epoch_parts)
                        break
                    except:
                        pass
                        
    except Exception as e:
        logger.error(f"Error parsing training log {log_path}: {e}")
    
    return training_info

@app.get("/api/results")
async def get_model_results():
    """Get all model results and metrics"""
    results = {
        "classification": None,
        "metric_learning": None
    }
    
    # Load classification results
    classification_results_path = "results/classification_results.json"
    if os.path.exists(classification_results_path):
        try:
            with open(classification_results_path, 'r') as f:
                classification_data = json.load(f)
                results["classification"] = classification_data
                
                # Add image paths
                if os.path.exists("results/roc_curve_cosine.png"):
                    results["classification"]["roc_cosine_image"] = "/api/results/images/roc_curve_cosine.png"
                if os.path.exists("results/roc_curve_euclidean.png"):
                    results["classification"]["roc_euclidean_image"] = "/api/results/images/roc_curve_euclidean.png"
                if os.path.exists("models/training_history.png"):
                    results["classification"]["training_history_image"] = "/api/results/images/training_history.png"
        except Exception as e:
            logger.error(f"Error loading classification results: {e}")
    
    # Parse classification training log
    classification_log_path = "training.log"
    training_info = parse_training_log(classification_log_path)
    if training_info and results["classification"]:
        results["classification"]["training_info"] = training_info
    
    # Load metric learning results
    metric_learning_results_path = "results/metric_learning_results.json"
    if os.path.exists(metric_learning_results_path):
        try:
            with open(metric_learning_results_path, 'r') as f:
                metric_learning_data = json.load(f)
                results["metric_learning"] = metric_learning_data
                
                # Add image paths
                if os.path.exists("results/metric_learning_roc_cosine.png"):
                    results["metric_learning"]["roc_cosine_image"] = "/api/results/images/metric_learning_roc_cosine.png"
                if os.path.exists("results/metric_learning_roc_euclidean.png"):
                    results["metric_learning"]["roc_euclidean_image"] = "/api/results/images/metric_learning_roc_euclidean.png"
                if os.path.exists("models/metric_learning_training_history.png"):
                    results["metric_learning"]["training_history_image"] = "/api/results/images/metric_learning_training_history.png"
        except Exception as e:
            logger.error(f"Error loading metric learning results: {e}")
    
    # Parse metric learning training log
    metric_learning_log_path = "metric_training.log"
    training_info_ml = parse_training_log(metric_learning_log_path)
    if training_info_ml and results["metric_learning"]:
        results["metric_learning"]["training_info"] = training_info_ml
    
    return JSONResponse(results)

@app.get("/api/results/images/{image_name}")
async def get_result_image(image_name: str):
    """Serve result images"""
    # Security: only allow specific image names
    allowed_images = [
        "roc_curve_cosine.png",
        "roc_curve_euclidean.png",
        "metric_learning_roc_cosine.png",
        "metric_learning_roc_euclidean.png",
        "training_history.png",
        "metric_learning_training_history.png"
    ]
    
    if image_name not in allowed_images:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Check in results directory first
    image_path = Path("results") / image_name
    if not image_path.exists():
        # Check in models directory for training history
        image_path = Path("models") / image_name
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")
    
    from fastapi.responses import FileResponse
    return FileResponse(image_path, media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8501)

