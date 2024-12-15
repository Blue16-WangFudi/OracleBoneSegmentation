# FileName:ODM——Oracle-Detect-Model

from ultralytics import YOLO
'''
调用yolov8进行模型预测（原文件放在images中）,并返回结果保存的地方(一般在runs文件夹下)
一个结果的目录结构：所有已经标记了的图片+对应图片的标签文件夹（labels）
'''
def detect(model_name,source):
    model = YOLO(model_name)
    # save保存预测可视化， save_txt保存预测
    model.predict(source=source, save_txt=True, save=True)
    return model.predictor.save_dir

def classify(model_name,source):
    model = YOLO(model_name)
    # save保存预测可视化， save_txt保存预测
    print(model.predict(source))
    return model.predictor.save_dir

if __name__ == '__main__':
    # print(detect('oraclechar_50.pt','ocr_out'))
    print(detect('oracle_28.pt', 'processed'))