import ultralytics
from ultralytics import YOLO

def main():
    # 加载YOLO模型配置文件  第8次
    model_path = 'runs/detect/train10/weights/last.pt'  # 模型配置文件路径----继续上次训练--使用上次的模型继续训练
    # model_path = 'datasets/yolov8.yaml'  # 模型配置文件路径
    data_path = 'datasets/data.yaml'  # 数据配置文件路径

    # 初始化模型
    model = YOLO(model_path)

    # 训练模型
    model.train(
        data=data_path,  # 数据配置文件路径
        epochs=5000,  # 训练周期数
        imgsz=1080,  # 输入图像大小
        resume=True,  # 是否从上次中断的地方继续训练
        workers=4,  # 数据加载工作线程数
        device='cuda',  # 训练设备，'cuda'表示使用GPU

        # 其他可能用到的参数，取消注释以使用
        batch=32,  # 每个批次的图像数量
        lr0=0.5,  # 初始学习率
        lrf=0.01,  # 最终学习率（初始学习率的比例）
        momentum=0.8,  # 优化器动量
        weight_decay=0.0005,  # 权重衰减
        # warmup_epochs=3.0,  # 预热周期数
        # warmup_bias_lr=0.1,  # 预热期间的学习率偏差
        # box=0.05,  # 边界框损失增益
        # cls=0.5,  # 分类损失增益
        # dfl=1.5,  # 分布聚焦损失增益
        # fl_gamma=0.0,  # Focal Loss Gamma
        # hsv_h=0.015,  # 色调增强范围
        # hsv_s=0.7,  # 饱和度增强范围
        # hsv_v=0.4,  # 明度增强范围
        # degrees=0.1,  # 图像旋转增强范围
        # translate=0.1,  # 图像平移增强范围
        # scale=0.5,  # 图像缩放增强范围
        # shear=0.0,  # 图像剪切增强范围
        # perspective=0.0,  # 图像透视增强范围
        # flipud=0.0,  # 上下翻转图像的概率
        # fliplr=0.5,  # 左右翻转图像的概率
        # mosaic=1.0,  # 马赛克增强概率
        # mixup=0.0,  # MixUp增强概率
        # copy_paste=0.0,  # Copy-paste增强概率
    )

if __name__ == '__main__':
    main()
