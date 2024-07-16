import time
import mss
import cv2
import numpy as np
from datetime import datetime
import os
import keyboard  # 用于监听按键事件

# 保存截图的目录
save_dir = 'D:/coco128/11/images'

# 如果目录不存在，则创建目录s
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 创建一个屏幕捕捉对象s
with mss.mss() as sct:
    print("Press 's' to take a screenshot, or 'q' to quit.")
    while True:
        # 检查是否按下了 's' 键进行截图
        if keyboard.is_pressed('s'):
            # 获取当前时间
            current_time = datetime.now()
            filename = current_time.strftime("%d%H%M%S%f")[:-4]  # 生成文件名，例如：1009013010

            # 获取屏幕截图
            sct_img = sct.grab(sct.monitors[0])  # 获取第一个显示器的信息

            # 将截图转换为OpenCV格式的图像
            img = np.array(sct_img)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # 将图像从BGRA转换为BGR格式

            # 构建保存路径
            save_path = os.path.join(save_dir, f"{filename}.png")

            # 保存图像
            cv2.imwrite(save_path, img)

            # 打印保存路径
            print(f"Saved screenshot: {save_path}")

            # 防止快速连续截图
            time.sleep(0.3)

        # 检查是否按下了 'q' 键退出
        if keyboard.is_pressed('q'):
            print("Exiting screenshot program.")
            break

