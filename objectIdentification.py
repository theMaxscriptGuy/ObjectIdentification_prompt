import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QLineEdit
)
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PySide6.QtCore import Qt
from ultralytics import YOLO
import cv2

class ObjectDetector(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Prompt - Object Finder")
        self.setGeometry(200, 200, 900, 700)

        # Layout
        layout = QVBoxLayout(self)

        # Widgets
        self.label = QLabel("Upload an image")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.clicked.connect(self.upload_image)
        layout.addWidget(self.upload_btn)

        self.prompt_box = QLineEdit()
        self.prompt_box.setPlaceholderText("Enter objects to find (e.g. bottle, vase, chair)")
        layout.addWidget(self.prompt_box)

        self.detect_btn = QPushButton("Find Objects")
        self.detect_btn.clicked.connect(self.find_objects)
        layout.addWidget(self.detect_btn)

        # Variables
        self.image_path = None
        self.model = YOLO("yolov8s.pt")  # small model

    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(self.image_path).scaled(800, 600, Qt.KeepAspectRatio)
            self.label.setPixmap(pixmap)

    def find_objects(self):
        if not self.image_path:
            self.label.setText("Please upload an image first.")
            return

        prompt_text = self.prompt_box.text().strip().lower()
        if not prompt_text:
            self.label.setText("Please enter one or more objects (comma separated).")
            return

        # Convert prompt to list of target objects
        target_objects = [p.strip() for p in prompt_text.split(",") if p.strip()]
        print("Looking for:", target_objects)

        # Run YOLO
        results = self.model(self.image_path)
        img = cv2.imread(self.image_path)
        h, w, ch = img.shape

        found_any = False
        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = self.model.names[cls].lower()
            if label in target_objects:  # check if detected label matches any requested
                found_any = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if not found_any:
            self.label.setText(f"No objects found from {target_objects}.")
            return

        # Convert OpenCV image (BGR) to QImage
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg).scaled(800, 600, Qt.KeepAspectRatio)
        self.label.setPixmap(pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ObjectDetector()
    window.show()
    sys.exit(app.exec())
