import sys
import cv2
import time
from ultralytics import YOLO
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog,
    QComboBox, QVBoxLayout, QWidget
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer


class YOLOApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("YOLO Detection App")
        self.setGeometry(100, 100, 800, 600)

        self.model_type_selector = QComboBox()
        self.model_type_selector.addItems(["Segmentation", "Pose Detection"])
        self.model_selector = QComboBox()
        self.source_selector = QComboBox()
        self.source_selector.addItems(["Webcam", "Image File", "Video File"])
        self.select_button = QPushButton("Select Source / Start")
        self.refresh_button = QPushButton("Refresh")
        self.save_frame_button = QPushButton("Save Current Frame")
        self.image_label = QLabel()

        layout = QVBoxLayout()
        layout.addWidget(self.model_type_selector)
        layout.addWidget(self.model_selector)
        layout.addWidget(self.source_selector)
        layout.addWidget(self.select_button)
        layout.addWidget(self.refresh_button)
        layout.addWidget(self.save_frame_button)
        layout.addWidget(self.image_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.model_type_selector.currentIndexChanged.connect(self.update_model_list)
        self.select_button.clicked.connect(self.start_detection)
        self.refresh_button.clicked.connect(self.reset_app)
        self.save_frame_button.clicked.connect(self.save_current_frame)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cap = None
        self.model = None
        self.task = None
        self.last_output_frame = None

        self.update_model_list()

    def reset_app(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None

        self.image_label.clear()

    def update_model_list(self):
        self.model_selector.clear()
        if self.model_type_selector.currentText() == "Segmentation":
            self.model_selector.addItems(["yolov8n-seg.pt", "yolov8m-seg.pt", "yolov8x-seg.pt"])
        else:
            self.model_selector.addItems(["yolov8-pose.pt"])

    def start_detection(self):
        model_path = self.model_selector.currentText()
        self.task = self.model_type_selector.currentText()
        self.model = YOLO(model_path)

        source_type = self.source_selector.currentText()

        if source_type == "Webcam":
            self.cap = cv2.VideoCapture(0)
            self.timer.start(30)
        elif source_type == "Image File":
            path, _ = QFileDialog.getOpenFileName(self, "Select Image")
            if path:
                frame = cv2.imread(path)
                self.display_result(frame)
        elif source_type == "Video File":
            path, _ = QFileDialog.getOpenFileName(self, "Select Video")
            if path:
                self.cap = cv2.VideoCapture(path)
                self.timer.start(30)

    def update_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return

        success, frame = self.cap.read()
        if not success:
            self.timer.stop()
            self.cap.release()
            return

        self.display_result(frame)

    def display_result(self, frame):
        start = time.perf_counter()
        results = self.model(frame)
        output_frame = results[0].plot()
        end = time.perf_counter()
        fps = 1 / (end - start)

        cv2.putText(output_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        rgb_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(self.image_label.size(), aspectRatioMode=1)
        self.image_label.setPixmap(pixmap)

        self.last_output_frame = output_frame.copy()

    def save_current_frame(self):
        if self.last_output_frame is not None:
            path, _ = QFileDialog.getSaveFileName(self, "Save Frame As", "", "PNG Files (*.png);;JPG Files (*.jpg)")
            if path:
                cv2.imwrite(path, self.last_output_frame)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLOApp()
    window.show()
    sys.exit(app.exec_())
