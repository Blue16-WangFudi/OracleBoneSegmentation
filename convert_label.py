# FileName:convert-label
import json
import os
import cv2
from PIL import Image


'''
将yolov8的label转为题目要求的格式(左上+右下坐标)
'''
def yolo_label_to_coordinate_label(image_filename, label_filename):
    # 获取图像width和height
    with Image.open(image_filename) as img:
        image_width, image_height = img.size

    # 读取标签文件
    with open(label_filename, 'r') as file:
        lines = file.readlines()

    converted_labels = []
    for line in lines:
        # 分隔每一条label
        parts = line.strip().split()
        class_id, center_x, center_y, width, height = parts

        # 文本转浮点数
        class_id = float(class_id)
        center_x = float(center_x) * image_width
        center_y = float(center_y) * image_height
        width = float(width) * image_width
        height = float(height) * image_height

        # 计算左上角坐标 (x, y)
        x = center_x - width / 2
        y = center_y - height / 2

        # 计算右下角坐标 (x, y)
        x2 = center_x + width / 2
        y2 = center_y + height / 2

        # 添加为list
        converted_labels.append([round(x,1), round(y,1), round(x2,1), round(y2,1), class_id])

    return converted_labels
'''
将题目数据集的label转为yolov8训练时需要的label格式
'''
def coordinate_label_to_yolo_label(json_filename):
    with open(os.path.join(json_filename), 'r') as f:
        data = json.load(f)
        annotations=data['ann']
        img_name= data['img_name']
    text=""

    image = cv2.imread(json_filename.replace('.json', '.jpg'))
    # 获取图像尺寸
    pic_height, pic_width, pic_channels = image.shape

    for annotation in annotations:
        # 注意参数需要正则化
        x_1=annotation[0]
        y_1 = annotation[1]
        x_2=annotation[2]
        y_2=annotation[3]
        class_id=int(annotation[4])
        center_x=(x_1+x_2)/2
        center_y=(y_1+y_2)/2
        width=x_2-x_1
        height=y_2-y_1
        text=text+str(f"{class_id} {center_x/pic_width} {center_y/pic_height} {width/pic_width} {height/pic_height}\n")
    # 读取到所有标注后，转换为yolo_txt格式(class center_x center_y width height)
    # {"img_ name": "w01906", "ann": [[111.0, 197.0, 152.0, 257.0, 1 .0], [108.0, 283.0, 157.0,
    with open("datasets/label/"+img_name+".txt", 'w', encoding='utf-8') as file:
        file.write(text)
    return img_name

if __name__ == '__main__':
    print(yolo_label_to_coordinate_label("runs/detect/predict/h02060.jpg", "runs/detect/predict/labels/h02060.txt"))