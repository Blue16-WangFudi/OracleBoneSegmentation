# FileName:convert-datasets
import os
import uuid
import cv2
import yaml
from shutil import copy
from convert_label import coordinate_label_to_yolo_label
from OPM import oracle_preprocess_model


def prepare_images(image_filename,img_name):
    # 读取图像：从文件中读取原始甲骨文图像
    img = cv2.imread(image_filename)

    # 预处理：调用预处理函数，处理原始图像
    processed = oracle_preprocess_model(img)

    # 显示结果：输出到指定数据集
    cv2.imwrite("datasets/image/"+img_name+".jpg", processed)  # 保存处理后的图像为'processed.jpg'

'''
转换所有2_Train的数据，输出到train_datasets下，以便yolo训练
'''
def convert_datasets_json(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否为JSON文件
        if filename.endswith('.json'):
            # 构建完整的文件路径
            file_path = os.path.join(folder_path, filename)
            # 调用prepare_labels函数处理JSON文件
            img_name = coordinate_label_to_yolo_label(file_path)
            # 构建原始图像的文件路径
            image_filename = os.path.join(folder_path, img_name + '.jpg')
            # 调用prepare_images函数处理图像
            prepare_images(image_filename, img_name)

def convert_datasets_folder(train_set_dir = 'trainSet',yaml_file_path = 'oraclechar/oraclechar.yaml',images_dir = 'oraclechar/images/train2017',labels_dir = 'oraclechar/labels/train2017'):
    # 确保目标目录存在
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # 获取目录下所有文件夹名称
    folders = os.listdir(train_set_dir)
    folders = [f for f in folders if os.path.isdir(os.path.join(train_set_dir, f))]

    # 更新yaml文件中的names配置项
    with open(yaml_file_path, 'r',encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 创建names字典
    names_dict = {i: folder for i, folder in enumerate(folders)}

    index_dict = {v: k for k, v in names_dict.items()}

    # 更新yaml文件
    config['names'] = names_dict

    with open(yaml_file_path, 'w') as f:
        yaml.safe_dump(config, f)

    # 遍历每个子目录
    for folder in folders:
        folder_path = os.path.join(train_set_dir, folder)
        files = os.listdir(folder_path)
        print(f"完成字：{folder}的处理，编号为{index_dict.get(folder)}")
        for file in files:
            # 重命名文件
            file_id = str(index_dict.get(folder))+"_"+str(uuid.uuid4())
            file_ext = os.path.splitext(file)[1]
            new_file_name = f"{file_id}{file_ext}"
            os.rename(os.path.join(folder_path, file), os.path.join(folder_path, new_file_name))

            # 复制图片到目标目录
            copy(os.path.join(folder_path, new_file_name), images_dir)

            img = cv2.imread(images_dir+"/"+new_file_name)
            processed = oracle_preprocess_model(img)
            cv2.imwrite(images_dir+"/"+new_file_name, processed)  # 保存处理后的图像

            # 生成标签文件
            label_content = f"{index_dict.get(folder)} 0.5 0.5 1 1"
            with open(os.path.join(labels_dir, f"{file_id}.txt"), 'w') as label_file:
                label_file.write(label_content)

    print("数据处理完成。")
if __name__ == '__main__':
    # img_name=prepare_labels("2_Train/b02520.json")
    prepare_images("images/w01870.jpg","w01870")
    # prepare_images("images/w01637.jpg","w01637")

    # convert_datasets("2_Train")