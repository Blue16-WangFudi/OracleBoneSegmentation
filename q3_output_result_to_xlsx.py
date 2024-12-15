import pandas as pd
import ODM
from convert_label import yolo_label_to_coordinate_label
import os
import cv2

from OPM import oracle_preprocess_model

'''
问题3：把数据放在images里面，运行后先将images中的要预测的数据转换并保存到processed中，然后调用模型预测，最后根据预测结果填写表格
'''

if __name__ == '__main__':
    for filename in os.listdir("images"):
        # 检查文件是否为pic文件
        if filename.endswith('.jpg'):
            img = cv2.imread("images/" + filename)
            processed = oracle_preprocess_model(img)
            cv2.imwrite("processed/" + filename, processed)  # 保存处理后的图像为'processed.jpg'

    result_path = detector.detect("oracle_28.pt", "processed/")
    table_path = "3_Test/Test_results.xlsx"
    # 思路：读取table_path文件中的第一列，然后遍历这一列每个单元格中的内容，根据内容（也就是文件名）在path/label中找到同文件名的txt文件，并将函数def convert_yolov8_to_required_format(image_filename, label_filename):的调用结果写到该单元格对应行的第二个格子中

    # 读取Excel文件
    df = pd.read_excel(table_path)

    # 获取第一列（文件名列）
    filenames = df.iloc[1:, 0]

    # 遍历文件名列
    for index, filename in enumerate(filenames,start=1):
        # 构建txt文件路径
        label_filename = os.path.join(result_path, 'labels\\'+filename.replace(".jpg",".txt"))
        # 检查txt文件是否存在
        if os.path.exists(label_filename):
            # 调用转换函数并获取结果
            result = yolo_label_to_coordinate_label(str(result_path) + "/" + filename, str(label_filename))

            # 将结果写入对应行的第二个格子
            df.iloc[index, 1] = str(result)[1:-1]
        else:
            print(f"File {label_filename} does not exist.")

    # 保存修改后的Excel文件
    df.to_excel(table_path, index=False)

