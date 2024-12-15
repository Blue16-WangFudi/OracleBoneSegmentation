# FileName:OOM——Oracle-Ocr-Model
import os
import cv2
import ODM

'''
通过标签分割图像并保存到与图片名称相同的子目录中，以便后续识别
'''
def pic_seperate_by_label():
    # 给定的目录A和目录B
    directory_A = "ocr_in"
    directory_B = "ocr_out"

    # 遍历目录A中的所有文件
    for filename in os.listdir(directory_A):
        # 检查文件是否为图像
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 构建图像和标签的完整路径
            image_path = os.path.join(directory_A, filename)
            label_path = os.path.join(directory_A, 'labels', filename.rsplit('.', 1)[0] + '.txt')

            # 读取图像
            image = cv2.imread(image_path)
            image_height, image_width = image.shape[:2]
            if os.path.exists(label_path):
                # 读取标签
                with open(label_path, 'r') as file:
                    annotations = file.readlines()

                # 为每个标注创建一个子目录
                subdirectory = os.path.join(directory_B, filename.rsplit('.', 1)[0])
                os.makedirs(subdirectory, exist_ok=True)

                # 遍历所有标注
                index = 0
                for annotation in annotations:
                    # 解析标注数据
                    class_id, x_center, y_center, width, height = map(float, annotation.split())

                    # 将比例转换为像素坐标
                    x_center_pixel = int(x_center * image_width)
                    y_center_pixel = int(y_center * image_height)
                    width_pixel = int(width * image_width)
                    height_pixel = int(height * image_height)

                    # 计算矩形框的坐标
                    x_start = max(x_center_pixel - width_pixel // 2, 0)
                    y_start = max(y_center_pixel - height_pixel // 2, 0)
                    x_end = x_start + width_pixel
                    y_end = y_start + height_pixel

                    # 裁剪图像
                    image_patch = image[y_start:y_end, x_start:x_end]

                    # 调用predict函数获取文件名
                    predicted_name = ODM.predict(image_patch)

                    # 保存裁剪后的图像
                    save_path = os.path.join(subdirectory, str(index) + '.jpg')
                    cv2.imwrite(save_path, image_patch)
                    index = index + 1
                    print(f"Saved cropped image to {save_path}")
                # 一个annotation完成后，直接执行识别
                # detector.classify('oraclechar.pt', subdirectory)

    print("Image cropping completed.")

'''
在已经分割的目录（一般是ocr_out）下，为每个甲骨文拓片的甲骨文分割片调用进行预测，返回读取到的目录列表（dirs），用于后续调用重命名方法做准备
'''
def traverse_and_detect(directory):
    # 遍历给定目录
    for root, dirs, files in os.walk(directory):
        # 对每个子目录调用detector的detect方法
        for dir in dirs:
            sub_dir_path = os.path.join(root, dir)
            ODM.detect('model/oraclechar_50.pt', sub_dir_path)
'''
# 真实目录名称列表示例（就是dirs中的内容）
real_names = [
    'w01790', 'w01791', 'w01792', 'w01793', 'w01794', 'w01795', 'W01796',
    'w01797', 'w01798', 'w01799', 'w01800', 'w01801', 'w01802', 'w01803',
    'w01804', 'w01806', 'w01807', 'w01808', 'W01809', 'w01810', 'w01811',
    'w01812', 'w01813', 'w01814', 'w01815', 'w01816', 'w01817', 'w01818',
    'w01819', 'w01820', 'w01821', 'w01822', 'w01823', 'w01825', 'w01826',
    'w01828', 'w01829', 'w01830', 'w01831', 'w01832', 'w01833', 'w01834',
    'w01835', 'w01836', 'w01837', 'w01838', 'w01839', 'w01840', 'w01841',
    'w01842'
]
'''
def result_rename(real_names):
    # 构建name_mapping字典，包括predict文件夹的特殊情况
    name_mapping = {1: real_names[0]}
    for i, name in enumerate(real_names[1:], start=2):
        name_mapping[i] = name

    def rename_subdirectories(directory, name_mapping):
        # 遍历给定目录
        for root, dirs, files in os.walk(directory):
            # 对每个子目录进行重命名
            for dir in dirs:
                # 如果子目录名是以predict开头的，那么进行重命名
                if dir.startswith('predict'):
                    # 获取predict后的数字
                    predict_number = int(dir.replace('predict', '')) if dir != 'predict' else 1
                    # 获取对应的真实目录名称
                    real_name = name_mapping.get(predict_number)
                    if real_name:
                        # 构建旧的子目录完整路径
                        old_dir_path = os.path.join(root, dir)
                        # 构建新的子目录完整路径
                        new_dir_path = os.path.join(root, real_name)
                        # 重命名子目录
                        os.rename(old_dir_path, new_dir_path)
                    else:
                        print(f"No real name found for directory: {dir}")