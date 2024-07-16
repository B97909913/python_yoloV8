import time
import mss
import numpy as np
import cv2
import ultralytics
from ultralytics import YOLO
from ultralytics.engine.results import Results


def chuli(result):
    # 假设 result 是 YOLOv8 的返回结果

    result1 = result[0]
    boxes = result1.boxes
    print(f"{boxes}boxes-----")
    print(f"{result1.orig_img}111111")
    print(f"{result1.orig_shape}22222")
    print(f"{result1.path}333")
    print(f"{result1.save_dir}44")
    print(f"{result1.speed}")
    # 提取原始图像
    orig_img = result1.orig_img
    print(f"orig_img{orig_img}推理结果")

    # 提取检测到的边界框信息
    boxeszuo = boxes.xyxy  # 获取边界框的坐标
    scores = result1.probs  # 获取置信度分数
    classes = boxes.cls  # 获取检测到的类别
    print(f"boxes{boxeszuo}推理结果1")
    print(f"scores{scores}推理结果2")
    print(f"classes{classes}推理结果3")
    # 提取类别名
    names = result1.names
    print(f"names----{names}推理结果4")
    # 在原始图像上绘制边界框

    # 确保 boxes, scores 和 classes 都不为空
    if boxeszuo is not None and scores is not None and classes is not None:
        for i, (box, score, cls) in enumerate(zip(boxeszuo, scores, classes)):
            # 获取边界框坐标
            x1, y1, x2, y2 = map(int, box)
            # 获取类别名称
            label = names[int(cls)]
            # 绘制边界框
            cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 绘制标签和置信度
            label_text = f'{label} {score:.2f}'
            # 显示图像
            cv2.putText(orig_img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow('Detection', orig_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print(f"{boxeszuo}-boxeszuo+{scores}-scores+{classes}-classes未检测到输入框------")




# 加载YOLO模型
model_path = 'runs/detect/train7/weights/best.pt'  # 修改为你的模型路径
model = YOLO(model_path)

# 创建一个屏幕捕捉对象
with mss.mss() as sct:
    # 定义捕捉区域
    monitor = sct.monitors[1]  # 获取第二个显示器的信息，根据需要调整

    while True:
        # 获取屏幕截图
        sct_img = sct.grab(monitor)

        # 将截图转换为OpenCV格式的图像
        img = np.array(sct_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # 将图像从BGRA转换为BGR格式

        # 进行YOLO推理
        results = model(img)
        print(f"推理结果{results}推理结果")

        # 处理推理结果并绘制到图像上
        if len(results) > 0:
            chuli(results)
        else:
            # 如果未检测到物体，打印消息并继续
            print("未检测到物体")

        # 等待2秒再进行下一次推理
        time.sleep(2)

        # 检测到按下ESC键时退出循环
        if cv2.waitKey(1) == 27:
            break

# 关闭所有窗口
cv2.destroyAllWindows()
