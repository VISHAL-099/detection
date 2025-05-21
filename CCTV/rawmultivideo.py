import sys
import json
import cv2
import numpy as np
import time
from collections import defaultdict
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from ultralytics import YOLO
import logging
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger('ultralytics').setLevel(logging.WARNING)

# Initialize YOLO model
model = YOLO('yolov8s.pt')

# Load label translations
with open('label_translation.json') as f:
    label_translation = json.load(f)

# Load URLs from json
with open('urls.json') as f:
    urls_dict = json.load(f)

frame_rate = 30

# Global variables for tracking object positions and stationary times
object_last_positions = defaultdict(tuple)
object_stationary_since = defaultdict(float)
stationary_threshold = 5  # 5 seconds threshold for stationary objects


def hash_to_color(label):
    hash_value = hash(label) % (256 * 256 * 256)
    b = hash_value % 256
    g = (hash_value // 256) % 256
    r = (hash_value // (256 * 256)) % 256
    return (b, g, r)


def process_frame(frame, start_time, alerts_callback):
    current_time = time.time() - start_time
    results = model(frame)
    detections = results[0].boxes.data.cpu().numpy()

    object_counts = defaultdict(int)
    current_positions = defaultdict(list)

    # Track stationary objects
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = model.names[int(cls)]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        current_positions[label].append((center_x, center_y))
        object_counts[label] += 1

        # Track positions and check for stationary objects
        last_position = object_last_positions[label]
        if last_position == (center_x, center_y):
            if label not in object_stationary_since:
                object_stationary_since[label] = current_time
            elif current_time - object_stationary_since[label] >= stationary_threshold:
                alert_msg = f"ALERT: {label_translation.get(label, label)} stationary for {stationary_threshold} seconds!"
                alerts_callback(alert_msg)
        else:
            object_stationary_since.pop(label, None)

        object_last_positions[label] = (center_x, center_y)

    # Draw bounding boxes and label texts
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = model.names[int(cls)]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = hash_to_color(label_translation.get(label, label))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        label_text = f"Yeh Hai {label_translation.get(label, label)}"
        cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return frame


class WorkerSignals(QObject):
    result = pyqtSignal(np.ndarray)


class VideoProcessingWorker(QRunnable):
    def __init__(self, url, start_time, alerts_callback):
        super(VideoProcessingWorker, self).__init__()
        self.url = url
        self.start_time = start_time
        self.signals = WorkerSignals()
        self.alerts_callback = alerts_callback
        self.stop_flag = False

    def run(self):
        try:
            cap = cv2.VideoCapture(self.url)
            if not cap.isOpened():
                print(f"Error opening video stream {self.url}")
                return

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            while not self.stop_flag:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (width, height))

                # Process frame with YOLO model
                processed_frame = process_frame(frame, self.start_time, self.alerts_callback)

                # Emit processed frame for display
                self.signals.result.emit(processed_frame)

                time.sleep(1 / fps)

        except Exception as e:
            print(f"Error occurred in processing video: {e}")

    def stop(self):
        self.stop_flag = True


class VideoWidget(QWidget):
    def __init__(self, parent=None):
        super(VideoWidget, self).__init__(parent)
        self.image = None

    def set_image(self, image):
        self.image = image
        self.update()

    def paintEvent(self, event):
        if self.image is not None:
            painter = QPainter(self)
            painter.drawImage(self.rect(), self.image)


class DetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("System Monitoring CCTV")
        self.setGeometry(100, 100, 1675, 875)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)

        # Video feed widget
        self.video_widget = VideoWidget()
        self.video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layout.addWidget(self.video_widget, stretch=3)

        # Side panel layout
        self.side_panel = QVBoxLayout()
        self.layout.addLayout(self.side_panel, stretch=1)

        # Label for the camera buttons box
        self.camera_buttons_label = QLabel("Camera List")
        self.side_panel.addWidget(self.camera_buttons_label)

        # Button list (camera selectors)
        self.button_list = QListWidget()
        self.side_panel.addWidget(self.button_list)

        # Label for the alert box
        self.alert_box_label = QLabel("Alerts")
        self.side_panel.addWidget(self.alert_box_label)

        # Alert box for displaying alerts
        self.alert_box = QTextEdit()
        self.alert_box.setReadOnly(True)
        self.side_panel.addWidget(self.alert_box)

        # Initialize camera buttons
        self.buttons = {}
        self.active_button = None
        for key in urls_dict:
            button = QPushButton(key)
            button.clicked.connect(self.create_button_handler(key))
            item = QListWidgetItem(self.button_list)
            item.setSizeHint(button.sizeHint())
            self.button_list.setItemWidget(item, button)
            self.buttons[key] = button

        # New button to show all video feeds
        self.show_all_button = QPushButton("Show All Feeds")
        self.show_all_button.clicked.connect(self.show_all_feeds)
        self.side_panel.addWidget(self.show_all_button)

        # Thread pool and video processing
        self.thread_pool = QThreadPool()
        self.current_frame = None
        self.start_time = None
        self.current_urls = []
        self.worker = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.setInterval(1000 // 30)

    def create_button_handler(self, key):
        def handler():
            if self.worker:
                self.worker.stop()
            self.current_urls = urls_dict[key]
            self.start_time = time.time()
            self.current_frame = None
            self.alert_box.clear()  # Clear previous alerts
            self.process_videos()

            if self.active_button:
                self.active_button.setStyleSheet("background-color: none")
            self.active_button = self.buttons[key]
            self.active_button.setStyleSheet("background-color: yellow")
        return handler

    def process_videos(self):
        if not self.current_urls:
            return

        self.worker = VideoProcessingWorker(self.current_urls[0], self.start_time, self.add_alert)
        self.worker.signals.result.connect(self.update_frame)
        self.thread_pool.start(self.worker)
        self.timer.start()

    def add_alert(self, alert_msg):
        self.alert_box.append(alert_msg)

    def update_frame(self, frame=None):
        if frame is not None:
            height, width, channels = frame.shape
            bytes_per_line = channels * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_BGR888)
            self.video_widget.set_image(q_image)

    def show_all_feeds(self):
        # Gather all video URLs and process them together
        self.current_urls = [url for key in urls_dict for url in urls_dict[key]]
        self.start_time = time.time()
        self.alert_box.clear()  # Clear previous alerts

        # Create video workers for each feed
        self.worker = None
        self.worker_list = []
        for url in self.current_urls:
            worker = VideoProcessingWorker(url, self.start_time, self.add_alert)
            worker.signals.result.connect(self.update_frame)
            self.worker_list.append(worker)

        # Start all workers to show raw feeds
        for worker in self.worker_list:
            self.thread_pool.start(worker)

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DetectionApp()
    window.show()
    sys.exit(app.exec_())
