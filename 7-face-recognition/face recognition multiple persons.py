#a sub folder called known_people, each person has a folder named with his/her name, 
#and within the folder are a few images of the person

#Automatically loads all images from a structured directory containing subfolders for each known individual.
#Encodes each image only if it successfully detects faces (checked by the presence of encodings).
#Compares detected faces against all known face encodings and assigns names accordingly.
#------------------------------------------------------------------

import face_recognition
import cv2
import os

# Base path to the folder containing known people
mypath="/Users/nghiatran/projects/ai-ml/ml/face-recognition"
basepath = mypath+"/known_people/"


# Load the image you want to check
test_image_path = mypath+"/test_facial_recognition.png"
image = face_recognition.load_image_file(test_image_path)

# Find all the faces and face encodings in the unknown image
face_locations = face_recognition.face_locations(image)
face_encodings = face_recognition.face_encodings(image, face_locations)

# Dictionary to hold the images and names of known faces
known_face_encodings = []
known_face_names = []

# Traverse the directory containing known people
for person_name in os.listdir(basepath):
    person_folder = os.path.join(basepath, person_name)
    if os.path.isdir(person_folder):
        # Read each image file in person's folder
        for image_file in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_file)
            if os.path.isfile(image_path):
                # Load image and create encoding
                person_image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(person_image)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(person_name)

# Initialize variables
face_names = []

# Compare face encodings against known faces
for face_encoding in face_encodings:
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"

    # Use the known face with the smallest distance to the new face
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = face_distances.argmin()
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    face_names.append(name)

# Use OpenCV to show the results
for (top, right, bottom, left), name in zip(face_locations, face_names):
    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

# Display the resulting image
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
