import cv2
import numpy as np
import imutils
from imutils.video import FPS
import time

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

# Object Detection Function (with confidence parameter)
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
            detected_objects.append((startX, startY, endX, endY, label, confidence, idx))

            # Draw the rectangle and label on the frame
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 1)

    # Movement detection: Compare the current frame with the previous frame
    movement_detected = {}
    if previous_frame is not None:
        frame_diff = cv2.absdiff(previous_frame, frame)
        gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)

        # Detect movement for each object individually
        for (startX, startY, endX, endY, label, confidence, idx) in detected_objects:
            object_roi = thresh[startY:endY, startX:endX]
            if np.sum(object_roi) > 0:
                movement_detected[label] = True
            else:
                movement_detected[label] = False

    return frame, detected_objects, movement_detected

# Main function for real-time webcam detection
def start_detection(use_gpu=False, idle_threshold=10):
    configure_gpu(net, use_gpu)
    cap = cv2.VideoCapture(0)  # Use webcam (0 is the default webcam)

    if not cap.isOpened():
        print("Webcam could not be accessed.")
        return

    fps = FPS().start()
    previous_frame = None

    # Store idle times for each object
    object_idle_times = {label: 0 for label in CLASSES}
    last_movement_time = {label: time.time() for label in CLASSES}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to retrieve frame from webcam.")
            break

        frame_resized = imutils.resize(frame, width=800)
        frame, detected_objects, movement_detected = detect_objects(frame_resized, previous_frame, confidence_level=0.5)

        # Update idle times for each object
        for (startX, startY, endX, endY, label, confidence, idx) in detected_objects:
            if label in movement_detected:
                if movement_detected[label]:  # If the object is moving, reset the idle time
                    object_idle_times[label] = 0
                else:
                    # Increment the idle time for the object
                    object_idle_times[label] += 1
                    if object_idle_times[label] >= idle_threshold:
                        # Trigger the alert if the object has been idle for more than the threshold
                        if time.time() - last_movement_time[label] >= idle_threshold:
                            print(f"ALERT: {label} is not moving for {idle_threshold} seconds!")
                            last_movement_time[label] = time.time()

        # Update previous frame for next iteration
        previous_frame = frame_resized

        # Display the resulting frame
        cv2.imshow("Object Detection", frame)

        # FPS update
        fps.update()

        # Press 'Esc' to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    fps.stop()

    print(f"Elapsed Time: {fps.elapsed():.2f} seconds")
    print(f"Approx. FPS: {fps.fps():.2f}")

# Call the function to start detection
start_detection(use_gpu=False, idle_threshold=10)  # Set idle time threshold to 10 seconds
