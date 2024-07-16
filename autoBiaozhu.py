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

    def update(self, boxes, scores, classes):
        filtered = [(tuple(box), score, cls) for box, score, cls in zip(boxes, scores, classes) if score >= 0.4]
        self.buffer.append(filtered)

    def get_consensus(self):
        if len(self.buffer) < self.buffer_size:
            return []

        count_dict = defaultdict(int)
        for detection in self.buffer:
            for box, _, cls in detection:
                key = (box, int(cls))
                count_dict[key] += 1

        consensus = [key[0] for key, count in count_dict.items() if count > self.buffer_size // 2]
        return consensus


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

        buffer = DetectionBuffer()

        for _ in range(5):
            results = model(img)

            # 获取预测框
            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy()  # 边界框
            scores = result.boxes.conf.cpu().numpy()  # 置信度
            classes = result.boxes.cls.cpu().numpy()  # 类别

            buffer.update(boxes, scores, classes)

        consensus_boxes = buffer.get_consensus()

        # 转换为JSON格式
        objects = []
        for box in consensus_boxes:
            x1, y1, x2, y2 = map(int, box)
            cls_idx = np.where(np.all(np.isclose(boxes, box, atol=1e-4), axis=1))[0]
            if len(cls_idx) > 0:
                cls = classes[cls_idx[0]]
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

        label_file = os.path.join(output_labels_path,
                                  img_file.replace('.jpg', '.json').replace('.jpeg', '.json').replace('.png', '.json'))
        with open(label_file, 'w') as f:
            json.dump(label_data, f, indent=4)
