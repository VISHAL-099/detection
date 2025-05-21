import cv2
import numpy as np
import imutils
from imutils.video import FPS

# Classes and colors used for detection
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor", "clock"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load the pre-trained neural network model
net = cv2.dnn.readNetFromCaffe('ssd_files/MobileNetSSD_deploy.prototxt', 'ssd_files/MobileNetSSD_deploy.caffemodel')

def configure_gpu(net, use_gpu=True):
    if use_gpu:
        print("[INFO] Setting preferable backend and target to CUDA...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def detect_objects(frame, confidence_level=0.5):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_level:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)

            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, COLORS[idx], 1)

    return frame

def test_stream():
    # Define the RTMP stream link here
    stream_url = 'rtmp://localhost/live/stream'
    
    configure_gpu(net)
    print("[INFO] Accessing RTMP video stream at:", stream_url)
    
    vs = cv2.VideoCapture(stream_url)
    fps = FPS().start()
    
    while True:
        ret, frame = vs.read()
        if not ret:
            print("[INFO] Failed to retrieve frame.")
            break
        
        frame = imutils.resize(frame, width=1920)
        frame = detect_objects(frame)
        frame = imutils.resize(frame, height=1080)
        cv2.imshow('Live Stream Detection', frame)
        
        if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
            break
        
        fps.update()
        
        if cv2.getWindowProperty('Live Stream Detection', cv2.WND_PROP_VISIBLE) < 1:
            break

    vs.release()
    cv2.destroyAllWindows()
    fps.stop()

    print("[INFO] Elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))

# Call the test_stream function to start processing the stream
test_stream()
