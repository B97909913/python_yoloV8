import os
import json

def convert_coco_to_yolo(coco_json_path, images_dir, output_dir):
    # 打开并读取COCO格式的JSON文件
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # 如果输出目录不存在，则创建它
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 创建一个字典，将图像ID映射到图像文件名
    image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
    # 创建一个字典，将类别ID映射到类别名称
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # 遍历所有标注
    for ann in coco_data['annotations']:
        # 获取标注对应的图像ID和类别ID
        image_id = ann['image_id']
        category_id = ann['category_id']
        # 获取边界框坐标
        bbox = ann['bbox']

        # 获取对应图像的文件名
        image_filename = image_id_to_filename[image_id]
        # 生成标签文件名（同图像文件名，但扩展名为.txt）
        label_filename = os.path.splitext(image_filename)[0] + '.txt'
        # 生成标签文件的完整路径
        label_filepath = os.path.join(output_dir, label_filename)

        # 获取边界框的x, y, 宽度和高度
        x, y, width, height = bbox
        # 获取图像的宽度和高度
        image_width = coco_data['images'][image_id-1]['width']
        image_height = coco_data['images'][image_id-1]['height']

        # 计算中心点坐标和归一化后的宽度和高度
        x_center = (x + width / 2) / image_width
        y_center = (y + height / 2) / image_height
        width /= image_width
        height /= image_height

        # 将转换后的结果写入标签文件
        with open(label_filepath, 'a') as f:
            f.write(f"{category_id} {x_center} {y_center} {width} {height}\n")

    print("COCO to YOLO conversion completed")

# 路径
coco_json_path = 'D:/coco128/2/Dataset_COCO_20240712_123652.json'  # COCO格式的JSON文件路径
images_dir = 'D:/coco128/4/3'  # 图像文件所在目录
output_dir = 'D:/coco128/labels'  # 输出的YOLO格式标签文件目录

# 转换
convert_coco_to_yolo(coco_json_path, images_dir, output_dir)
