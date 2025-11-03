from ultralytics import YOLO 
import cv2 #pip install OpenCV-Python 

model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)


def main():
    #oneframe()
    camshow()

def camshow():
    cap = cv2.VideoCapture(0)  # '0' is typically the default value for the primary webcam

    if not cap.isOpened():
        print("Error: Could not open webcam")
        exit()

    while True:
        # Capture frame-by-frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        results = model(frame)        
        # Since we're processing one frame at a time, we'll access the first result
        result = results[0]
        print(f"Detected {len(result.boxes)} objects")

        if result.boxes:
            print("there are boxes")
            for box in result.boxes:
                coordinates = box.xyxy[0].tolist()  # Access the first element and convert to list
                if len(coordinates) == 4:
                    x1, y1, x2, y2 = coordinates
                    conf = box.conf.tolist()[0]  # Access confidence; assume it's accessible directly
                    cls = box.cls.tolist()[0]  # Access class index; assume it's accessible directly
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f'{result.names[int(cls)]} {conf:.2f}'
                    cv2.putText(frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    print("Unexpected number of coordinates:", coordinates)
                
        cv2.imshow('YOLOv8 Webcam Real-Time Detection', frame)        
        flagkey=waitforkey()  #press q to exit , press space to continue to next frame
        if flagkey=='exit':  break  

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def waitforkey():
        flagkey='not set'
        # Wait for the user to press the space bar to continue or 'q' to quit
        while True:
            key = cv2.waitKey(0)  # 0 means wait indefinitely for a key press
            print("key=",key)
            if key == ord('q'):
                flagkey='exit'
                return flagkey
            elif key == ord(' '):
                 flagkey='space'
                 return flagkey
            else:
                continue  # If any key other than space is pressed, re-loop without capturing a new frame


import os
def current_path():
    current_path = os.getcwd()
    print(current_path)
    
def oneframe():
    #results = model([mypath+'WeChat Image_20240414223445.jpg'])  # return a list of Results objects
    results = model([mypath+'car testing.png'])  # return a list of Results objects

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        result.show()  # display to screen
        result.save(filename='result.jpg')  # save to disk

main()