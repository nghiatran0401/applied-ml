"""
Streamlit User Interface for Face Recognition Attendance System
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import json
from pathlib import Path

# Import our modules
from src.utils.face_detector import FaceDetector
from src.utils.face_database import FaceDatabase
from src.modules.anti_spoofing import create_anti_spoofing_detector
from src.modules.emotion_detection import create_emotion_detector
from src.models.classification_model import create_model
from src.models.metric_learning_model import create_embedding_model


# Page configuration
st.set_page_config(
    page_title="Face Recognition Attendance System",
    page_icon="👤",
    layout="wide"
)

# Initialize session state
if 'database' not in st.session_state:
    st.session_state.database = None
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False


def load_models():
    """Load face recognition models"""
    if st.session_state.models_loaded:
        return
    
    try:
        # Check if models exist
        classification_model_path = "models/classification_best.pth"
        
        if not os.path.exists(classification_model_path):
            st.warning("⚠️ Classification model not found. Please train the model first.")
            return
        
        # Set device
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        
        st.session_state.models_loaded = True
        st.session_state.device = device
        
    except Exception as e:
        st.error(f"Error loading models: {e}")


def initialize_database():
    """Initialize face database"""
    if st.session_state.database is not None:
        return st.session_state.database
    
    # Check if database file exists
    db_path = "face_database.pkl"
    if os.path.exists(db_path):
        try:
            db = FaceDatabase(
                model_path="models/classification_best.pth",
                num_classes=4000,  # This should match your trained model
                device=st.session_state.get('device', 'cpu')
            )
            db.load(db_path)
            st.session_state.database = db
            return db
        except Exception as e:
            st.warning(f"Could not load existing database: {e}")
    
    # Create new database
    db = FaceDatabase(
        model_path="models/classification_best.pth",
        num_classes=4000,
        device=st.session_state.get('device', 'cpu')
    )
    st.session_state.database = db
    return db


def main():
    """Main application"""
    st.title("👤 Face Recognition Attendance System")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Mode",
        ["🏠 Home", "📝 Registration", "✅ Verification", "📊 Database"]
    )
    
    # Load models
    load_models()
    
    if page == "🏠 Home":
        show_home()
    elif page == "📝 Registration":
        show_registration()
    elif page == "✅ Verification":
        show_verification()
    elif page == "📊 Database":
        show_database()


def show_home():
    """Home page"""
    st.header("Welcome to Face Recognition Attendance System")
    
    st.markdown("""
    ### Features:
    - **Face Verification**: Two approaches (Classification & Metric Learning)
    - **Anti-Spoofing**: Detect fake faces (printed photos, screen photos)
    - **Emotion Detection**: Identify emotional states
    - **User Interface**: Easy registration and verification
    
    ### How to Use:
    1. **Registration**: Register new employees by uploading their photos
    2. **Verification**: Verify if a person is registered in the system
    3. **Database**: View registered people
    
    ### Requirements:
    - Trained models (classification and/or metric learning)
    - Face database (created during registration)
    """)
    
    # Check system status
    st.markdown("### System Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if os.path.exists("models/classification_best.pth"):
            st.success("✅ Classification Model")
        else:
            st.error("❌ Classification Model")
    
    with col2:
        if os.path.exists("models/metric_learning_best.pth"):
            st.success("✅ Metric Learning Model")
        else:
            st.warning("⚠️ Metric Learning Model")
    
    with col3:
        if st.session_state.database is not None:
            st.success("✅ Face Database")
        else:
            st.info("ℹ️ Face Database (empty)")


def show_registration():
    """Registration page"""
    st.header("📝 Register New Person")
    
    # Initialize database
    db = initialize_database()
    
    # Input form
    with st.form("registration_form"):
        person_name = st.text_input("Person Name", placeholder="Enter person's name")
        
        # Image upload
        uploaded_files = st.file_uploader(
            "Upload Face Images",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Upload multiple images of the same person for better accuracy"
        )
        
        submitted = st.form_submit_button("Register Person")
        
        if submitted:
            if not person_name:
                st.error("Please enter a person name")
                return
            
            if not uploaded_files:
                st.error("Please upload at least one image")
                return
            
            # Process images
            image_paths = []
            temp_dir = Path("temp_uploads")
            temp_dir.mkdir(exist_ok=True)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing image {i+1}/{len(uploaded_files)}...")
                
                # Save uploaded file
                image_path = temp_dir / uploaded_file.name
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                image_paths.append(str(image_path))
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            # Register person
            status_text.text("Registering person in database...")
            try:
                person_id = len(db.database)
                db.register_person(person_id, person_name, image_paths)
                
                # Save database
                db.save("face_database.pkl")
                
                # Clean up temp files
                for path in image_paths:
                    os.remove(path)
                
                st.success(f"✅ Successfully registered {person_name}!")
                st.balloons()
                
            except Exception as e:
                st.error(f"Error during registration: {e}")
            
            progress_bar.empty()
            status_text.empty()


def show_verification():
    """Verification page"""
    st.header("✅ Verify Person")
    
    # Initialize components
    db = initialize_database()
    face_detector = FaceDetector()
    anti_spoofing = create_anti_spoofing_detector(method='heuristic')
    emotion_detector = create_emotion_detector(method='fer')
    
    # Input method
    input_method = st.radio(
        "Input Method",
        ["📷 Upload Image", "📸 Camera"]
    )
    
    image = None
    
    if input_method == "📷 Upload Image":
        uploaded_file = st.file_uploader(
            "Upload Face Image",
            type=['jpg', 'jpeg', 'png']
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
    
    elif input_method == "📸 Camera":
        camera_image = st.camera_input("Take a photo")
        if camera_image is not None:
            image = Image.open(camera_image).convert('RGB')
    
    if image is not None:
        # Display image
        st.image(image, caption="Input Image", use_container_width=True)
        
        # Process button
        if st.button("🔍 Verify", type="primary"):
            with st.spinner("Processing..."):
                # Step 1: Face Detection
                st.markdown("### Step 1: Face Detection")
                try:
                    # For now, use the image path if available
                    # In production, you'd process the PIL image directly
                    temp_path = "temp_verify.jpg"
                    image.save(temp_path)
                    
                    face, bbox = face_detector.detect_and_align(temp_path)
                    if face is None:
                        st.error("❌ No face detected in image")
                        os.remove(temp_path)
                        return
                    
                    st.success("✅ Face detected")
                    os.remove(temp_path)
                    
                except Exception as e:
                    st.error(f"Face detection error: {e}")
                    return
                
                # Step 2: Anti-Spoofing
                st.markdown("### Step 2: Anti-Spoofing Check")
                is_real, liveness_confidence = anti_spoofing.detect(image)
                
                if is_real:
                    st.success(f"✅ Real Face (Confidence: {liveness_confidence:.2%})")
                else:
                    st.error(f"❌ Spoofed Face Detected (Confidence: {liveness_confidence:.2%})")
                    st.warning("⚠️ This may be a fake face (printed photo or screen)")
                
                # Step 3: Emotion Detection
                st.markdown("### Step 3: Emotion Detection")
                emotion, emotion_confidence, all_emotions = emotion_detector.detect(image)
                emotion_icon = emotion_detector.get_emotion_icon(emotion)
                
                st.markdown(f"**Detected Emotion:** {emotion_icon} {emotion.capitalize()} ({emotion_confidence:.2%})")
                
                # Show all emotions
                with st.expander("View All Emotion Scores"):
                    for emo, score in all_emotions.items():
                        st.progress(score, text=f"{emo.capitalize()}: {score:.2%}")
                
                # Step 4: Face Recognition (only if real face)
                if is_real:
                    st.markdown("### Step 4: Face Recognition")
                    
                    if len(db.database) == 0:
                        st.warning("⚠️ No registered people in database. Please register someone first.")
                    else:
                        try:
                            # Extract embedding
                            embedding = db.extract_embedding(temp_path if os.path.exists("temp_verify.jpg") else None)
                            
                            if embedding is None:
                                st.error("Could not extract face embedding")
                            else:
                                # Find match
                                person_id, person_name, similarity = db.find_match(
                                    embedding, threshold=0.6, metric='cosine'
                                )
                                
                                if person_id is not None:
                                    st.success(f"✅ **Match Found!**")
                                    st.markdown(f"**Name:** {person_name}")
                                    st.markdown(f"**Similarity:** {similarity:.2%}")
                                    st.balloons()
                                else:
                                    st.info("ℹ️ **No Match Found**")
                                    st.markdown(f"Best similarity: {similarity:.2%} (below threshold)")
                                    st.markdown("This person is not registered in the system.")
                        
                        except Exception as e:
                            st.error(f"Face recognition error: {e}")
                else:
                    st.warning("⚠️ Skipping face recognition due to spoofed face detection")


def show_database():
    """Database page"""
    st.header("📊 Registered People Database")
    
    db = initialize_database()
    
    if len(db.database) == 0:
        st.info("No people registered yet. Go to Registration page to add people.")
    else:
        st.markdown(f"**Total Registered:** {len(db.database)} people")
        
        # Display registered people
        for person_id, person_name in db.person_names.items():
            with st.expander(f"👤 {person_name} (ID: {person_id})"):
                st.write(f"**Number of face embeddings:** {len(db.database[person_id])}")
                
                if st.button(f"Delete {person_name}", key=f"delete_{person_id}"):
                    # Delete person
                    del db.database[person_id]
                    del db.person_names[person_id]
                    db.save("face_database.pkl")
                    st.success(f"Deleted {person_name}")
                    st.experimental_rerun()


if __name__ == "__main__":
    main()

