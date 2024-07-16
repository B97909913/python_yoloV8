import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import json
import time

class DetectionBuffer:
    def __init__(self, buffer_size=5):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.processed_overlaps = set()  # Track processed overlapping boxes

    def update(self, boxes, scores, classes):
        filtered = [(tuple(box), score, cls) for box, score, cls in zip(boxes, scores, classes) if score >= 0.4]
        self.buffer.append(filtered)

    def get_overlapping_boxes(self):
        if len(self.buffer) < self.buffer_size:
            return []

        count_dict = defaultdict(list)
        for detection in self.buffer:
            for box, score, cls in detection:
                count_dict[cls].append(box)

        overlapping_boxes = []
        for cls, boxes_list in count_dict.items():
            if len(boxes_list) >= 2:  # Only consider classes that appear 2 or more times
                for i in range(len(boxes_list) - 1):
                    for j in range(i + 1, len(boxes_list)):
                        box1 = boxes_list[i]
                        box2 = boxes_list[j]
                        if self.is_overlapping(box1, box2):
                            overlapping_box = self.get_overlap(box1, box2)
                            if overlapping_box not in self.processed_overlaps:
                                self.processed_overlaps.add(overlapping_box)
                                overlapping_boxes.append((*overlapping_box, cls))

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

# 加载模型
model_path = 'runs/detect/train8/weights/best.pt'  # 修改为你的模型路径
model = YOLO(model_path)

# 新数据集路径
new_images_path = 'D:/coco128/10/images'
output_labels_path = 'D:/coco128/10/labels'

if not os.path.exists(output_labels_path):
    os.makedirs(output_labels_path)

# 自动标注新数据集
for img_file in os.listdir(new_images_path):
    if img_file.endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(new_images_path, img_file)
        img = cv2.imread(img_path)

        buffer = DetectionBuffer(buffer_size=5)

        # 进行5次检测
        for _ in range(5):
            results = model(img)

            # 获取预测框
            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy()  # 边界框
            scores = result.boxes.conf.cpu().numpy()  # 置信度
            classes = result.boxes.cls.cpu().numpy()  # 类别

            buffer.update(boxes, scores, classes)

        overlapping_boxes = buffer.get_overlapping_boxes()

        # 转换为JSON格式
        objects = []
        for box in overlapping_boxes:
            x1, y1, x2, y2, cls = map(int, box)
            label = result.names[int(cls)]
            objects.append({
                "name": label,
                "bndbox": {
                    "xmin": x1,
                    "ymin": y1,
                    "xmax": x2,
                    "ymax": y2
                }
            })

        label_data = {
            "path": img_path,
            "outputs": {"object": objects},
            "time_labeled": int(time.time() * 1000),
            "labeled": True,
            "size": {
                "width": img.shape[1],
                "height": img.shape[0],
                "depth": img.shape[2]
            }
        }

        label_file = os.path.join(output_labels_path, img_file.replace('.jpg', '.json').replace('.jpeg', '.json').replace('.png', '.json'))
        with open(label_file, 'w') as f:
            json.dump(label_data, f, indent=4)
