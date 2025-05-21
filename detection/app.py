import cv2
import numpy as np
import streamlit as st
from imutils.video import FPS
import imutils

# Define Classes and Colors
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor", "clock"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load the pre-trained model
net = cv2.dnn.readNetFromCaffe('ssd_files/MobileNetSSD_deploy.prototxt', 'ssd_files/MobileNetSSD_deploy.caffemodel')

# Configure GPU/CPU
def configure_gpu(net, use_gpu=True):
    if use_gpu:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Object Detection Function
def detect_objects(frame, previous_frame, confidence_level=0.5):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    detected_objects = []
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_level:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            detected_objects.append((startX, startY, endX, endY, label, confidence))

            # Draw the rectangle and label on the frame
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 1)
    
    # Movement detection: Compare the current frame with the previous frame
    movement_detected = False
    if previous_frame is not None:
        frame_diff = cv2.absdiff(previous_frame, frame)
        gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)
        movement_detected = np.sum(thresh) > 0

    return frame, detected_objects, movement_detected

# Streamlit UI
st.title("Real-Time Object Detection with Webcam")
st.sidebar.title("Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

use_gpu = st.sidebar.checkbox("Use GPU", value=False)
configure_gpu(net, use_gpu)

# Placeholder for the alert
alert_placeholder = st.empty()

# Live Webcam Feed
if st.button("Start Detection", key="start_button"):
    stframe = st.empty()  # Placeholder for the live stream
    cap = cv2.VideoCapture(0)  # Use webcam (0 is the default webcam)
    
    if not cap.isOpened():
        st.error("Webcam could not be accessed.")
    else:
        fps = FPS().start()
        previous_frame = None
        stop_detection = False

        while not stop_detection:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to retrieve frame from webcam.")
                break

            frame_resized = imutils.resize(frame, width=800)
            frame, detected_objects, movement_detected = detect_objects(frame_resized, previous_frame, confidence)

            # Update previous frame for next iteration
            previous_frame = frame_resized

            # Convert frame to RGB (for Streamlit)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", use_column_width=True)
            
            fps.update()

            # Movement alert
            if not movement_detected:
                alert_placeholder.warning("No movement detected. Please check if the object is moving!")

            # Stop detection button (only one button)
            stop_detection = st.button("Stop Detection", key="stop_button")
        
        cap.release()
        fps.stop()
        st.write(f"Elapsed Time: {fps.elapsed():.2f} seconds")
        st.write(f"Approx. FPS: {fps.fps():.2f}")
