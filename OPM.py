# FileName:OPM——Oracle-Preprocess-Model
import cv2
'''
OPM模型——图像预处理
'''
def oracle_preprocess_model(img):
    # 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahed = clahe.apply(gray)

    # 二值化
    _, binary = cv2.threshold(clahed, 150, 255, cv2.THRESH_BINARY_INV)

    # 去噪
    denoised = cv2.medianBlur(binary, 3)

    # 锐化
    sharpened = cv2.GaussianBlur(denoised, (0, 0), 3)
    sharpened = cv2.addWeighted(denoised, 1.5, sharpened, -0.5, 0)

    return sharpened