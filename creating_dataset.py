import os
import shutil
import numpy as np


img_path = 'D:/PycharmProject/tumor_detection/brain tumor/brain_tumor_dataset/'

# split the data by train/val/test
for CLASS in os.listdir(img_path):
    if not CLASS.startswith('.'):
        IMG_NUM = len(os.listdir(img_path + CLASS))
        for (n, FILE_NAME) in enumerate(os.listdir(img_path + CLASS)):
            img = img_path + CLASS + '/' + FILE_NAME
            if n < 300:
                shutil.copy(src=img, dst=img_path + '.b/TEST/' + CLASS.upper() + '/' + FILE_NAME)
            
            elif n < 0.8 * IMG_NUM:
                shutil.copy(src=img, dst=img_path + '.b/TRAIN/' + CLASS.upper() + '/' + FILE_NAME)
            
            else:
                shutil.copy(src=img, dst=img_path + '.b/VAL/' + CLASS.upper() + '/' + FILE_NAME)
