import sys
import time
import mss
import numpy as np
import cv2
from ultralytics import YOLO
from PyQt5 import QtWidgets, QtGui, QtCore
from collections import defaultdict, deque


class OverlayWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowTransparentForInput)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.screen = QtWidgets.QApplication.primaryScreen()
        screen_geometry = self.screen.geometry()
        self.setGeometry(screen_geometry)
        self.image = None

    def set_image(self, img):
        self.image = img
        self.update()

    def paintEvent(self, event):
        if self.image is not None:
            painter = QtGui.QPainter(self)
            pixmap = QtGui.QPixmap.fromImage(self.image)
            painter.drawPixmap(self.rect(), pixmap)
        super().paintEvent(event)


class DetectionBuffer:
    def __init__(self, buffer_size=3):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def update(self, boxes, scores, classes):
        filtered = []
        for box, score, cls in zip(boxes, scores, classes):
            if score >= 0.01:
                filtered.append((tuple(box), score, cls))
        self.buffer.append(filtered)

    def get_overlapping_boxes(self):
        if len(self.buffer) < 2:
            return []

        overlapping_boxes = []
        for i in range(len(self.buffer) - 1):
            current_frame = self.buffer[i]
            next_frame = self.buffer[i + 1]
            for box1, score1, cls1 in current_frame:
                for box2, score2, cls2 in next_frame:
                    if cls1 == cls2 and self.is_overlapping(box1, box2):
                        overlapping_box = self.get_overlap(box1, box2)
                        overlapping_boxes.append((*overlapping_box, cls1))
        return overlapping_boxes

    def is_overlapping(self, box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        return not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)

    def get_overlap(self, box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        overlap_x_min = max(x1_min, x2_min)
        overlap_y_min = max(y1_min, y2_min)
        overlap_x_max = min(x1_max, x2_max)
        overlap_y_max = min(y1_max, y2_max)
        return overlap_x_min, overlap_y_min, overlap_x_max, overlap_y_max


def chuli(result, orig_img, buffer):
    result1 = result[0]
    boxes = result1.boxes.xyxy.cpu().numpy()
    scores = result1.boxes.conf.cpu().numpy()
    classes = result1.boxes.cls.cpu().numpy()
    names = result1.names

    buffer.update(boxes, scores, classes)
    overlapping_boxes = buffer.get_overlapping_boxes()

    overlay_img = np.zeros_like(orig_img, dtype=np.uint8)

    for box in overlapping_boxes:
        x1, y1, x2, y2, cls = map(int, box)
        if cls < len(names):
            label = names[cls]
            cv2.rectangle(overlay_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f'{label}'
            cv2.putText(overlay_img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    height, width, channel = overlay_img.shape
    bytes_per_line = 3 * width
    qt_image = QtGui.QImage(overlay_img.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
    return qt_image.rgbSwapped()


model_path = 'runs/detect/train10/weights/best.pt'
model = YOLO(model_path)

with mss.mss() as sct:
    app = QtWidgets.QApplication(sys.argv)
    overlay = OverlayWindow()
    overlay.setWindowOpacity(0.6)
    overlay.show()

    monitor = {"top": 0, "left": 0, "width": overlay.screen.geometry().width(), "height": overlay.screen.geometry().height()}

    buffer = DetectionBuffer(buffer_size=3)

    while True:
        sct_img = sct.grab(monitor)
        img = np.array(sct_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        results = model(img)

        if len(results) > 0:
            qt_img = chuli(results, img.copy(), buffer)
            overlay.set_image(qt_img)
        else:
            overlay.set_image(None)

        app.processEvents()

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    sys.exit(app.exec_())
