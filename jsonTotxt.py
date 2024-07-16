import os
import json

# 字典来存储类别ID映射
class_id_mapping = {}

# 从文件中加载类别ID映射
def load_class_id_mapping(mapping_file):
    global class_id_mapping
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r', encoding='utf-8') as f:
            data = f.read()
            if data.startswith("names:"):
                lines = data.split("\n")[1:]
                for line in lines:
                    if line.strip():
                        class_id, class_name = line.split(":")
                        class_id = int(class_id.strip())
                        class_name = class_name.strip()
                        class_id_mapping[class_name] = class_id

# 获取类别ID，如果类别名不在字典中则添加
def get_class_id(class_name):
    if class_name not in class_id_mapping:
        class_id_mapping[class_name] = len(class_id_mapping)
    return class_id_mapping[class_name]

# 将JSON格式的标注转换为YOLO格式
def convert_to_yolo_format(json_path, output_dir):
    # 打开并读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    width = data['size']['width']  # 获取图像宽度
    height = data['size']['height']  # 获取图像高度

    annotations = data['outputs']['object']  # 获取标注对象
    yolo_annotations = []

    for ann in annotations:
        class_name = ann['name'].strip()  # 获取类别名并去除前后空格
        class_id = get_class_id(class_name)  # 获取类别ID

        # 获取边界框的坐标
        xmin = ann['bndbox']['xmin']
        ymin = ann['bndbox']['ymin']
        xmax = ann['bndbox']['xmax']
        ymax = ann['bndbox']['ymax']

        # 将边界框坐标转换为YOLO格式
        x_center = (xmin + xmax) / 2 / width
        y_center = (ymin + ymax) / 2 / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height

        # 将YOLO格式的标注添加到列表中
        yolo_annotations.append(f"{class_id} {x_center} {y_center} {w} {h}\n")

    # 构建输出文件路径
    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(json_path))[0] + '.txt')
    # 将YOLO格式的标注写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(yolo_annotations)

# 保存类别ID映射到文件
def save_class_id_mapping(mapping_file):
    with open(mapping_file, 'w', encoding='utf-8') as f:
        f.write("names:\n")
        # 按照类别ID排序并写入文件
        for class_name, class_id in sorted(class_id_mapping.items(), key=lambda item: item[1]):
            f.write(f"    {class_id}: {class_name}\n")  # 这里决定了序号在前，类别名在后

# 处理目录中的所有JSON文件
def process_directory(input_dir, output_dir, mapping_file):
    # 如果输出目录不存在，则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载已有的类别ID映射
    load_class_id_mapping(mapping_file)

    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            json_path = os.path.join(input_dir, filename)  # 构建JSON文件路径
            convert_to_yolo_format(json_path, output_dir)  # 转换为YOLO格式

    # 保存类别ID映射到文件
    save_class_id_mapping(mapping_file)

# 指定输入目录，输出目录和映射文件路径
input_dir = 'D:/coco128/10/outputs'  # JSON标注文件所在的目录路径
output_dir = 'D:/coco128/10/txt'  # 保存转换后TXT文件的目录路径
mapping_file = 'D:/coco128/class_id_mapping.json'  # 保存类别ID映射的文件路径

# 处理目录中的所有JSON文件并保存类别ID映射
process_directory(input_dir, output_dir, mapping_file)
