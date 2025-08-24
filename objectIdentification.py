import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QVBoxLayout, QWidget, QFileDialog, QLineEdit
)
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PySide6.QtCore import Qt
from ultralytics import YOLO
import cv2

class ObjectDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Object Detection with Prompt")
        self.setGeometry(200, 200, 900, 700)

        # YOLO model
        self.model = YOLO("yolov8n.pt")  # use yolov8n.pt for speed, or yolov8s.pt for more accuracy
        self.image = None
        self.cv_image = None

        # Widgets
        self.label = QLabel("Load an image to start", self)
        self.label.setAlignment(Qt.AlignCenter)

        self.load_btn = QPushButton("Load Image", self)
        self.load_btn.clicked.connect(self.load_image)

        self.prompt_box = QLineEdit(self)
        self.prompt_box.setPlaceholderText("Type an object to detect (e.g. 'bottle', 'person')")

        self.detect_btn = QPushButton("Detect", self)
        self.detect_btn.clicked.connect(self.detect_objects)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.load_btn)
        layout.addWidget(self.prompt_box)
        layout.addWidget(self.detect_btn)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_name:
            self.cv_image = cv2.imread(file_name)
            self.display_image(self.cv_image)

    def display_image(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.label.setPixmap(pixmap.scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio))

    def detect_objects(self):
        if self.cv_image is None:
            self.label.setText("Please load an image first.")
            return

        target = self.prompt_box.text().strip().lower()
        if not target:
            self.label.setText("Please type an object to detect.")
            return

        results = self.model(self.cv_image)
        found = False
        img_with_boxes = self.cv_image.copy()

        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = self.model.names[cls].lower()
            if target in label:  # match user prompt
                found = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # Draw bounding box
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_with_boxes, f"{label} {conf:.2f}", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.9, (0, 255, 0), 2)

        if found:
            self.display_image(img_with_boxes)
        else:
            self.label.setText(f"No '{target}' found in image.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ObjectDetectionApp()
    window.show()
    sys.exit(app.exec())
