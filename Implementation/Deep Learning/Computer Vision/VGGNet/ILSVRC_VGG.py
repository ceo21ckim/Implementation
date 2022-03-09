# !pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

import torch
import json
import os
import numpy as np 
import urllib.request
import zipfile
from PIL import Image 
import torchvision
from torchvision import models, transforms
import matplotlib.pyplot as plt


# data 폴더가 존재하지 않는 경우 작성한다
data_dir = "./data/"
if not os.path.exists(data_dir):
    os.mkdir(data_dir)



# ImageNet의 class_index를 다운로드한다
# Keras에서 제공하는 항목
# https://github.com/fchollet/deep-learning-models/blob/master/imagenet_utils.py

url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
save_path = os.path.join(data_dir, "imagenet_class_index.json")

if not os.path.exists(save_path):
    urllib.request.urlretrieve(url, save_path)


# 1.3절에서 사용하는 개미와 벌의 화상 데이터를 다운로드하여 압축을 해제한다
# PyTorch의 튜토리얼로 제공되는 항목
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

url = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"
save_path = os.path.join(data_dir, "hymenoptera_data.zip")

if not os.path.exists(save_path):
    urllib.request.urlretrieve(url, save_path)

    # ZIP 파일을 읽는다
    zip = zipfile.ZipFile(save_path)
    zip.extractall(data_dir)  # ZIP을 압축 해제
    zip.close()  # ZIP 파일을 닫는다

    # ZIP 파일을 삭제
    os.remove(save_path)


net = models.vgg16(pretrained=True)


class BaseTransform():
    def __init__(self, resize, mean, std):
        self.base_transform = transforms.Compose([
            transforms.Resize(resize), # 짧은 변의 길이가 resize가 된다.
            transforms.CenterCrop(resize), # 이미지 중앙을 resize x resize 로 자른다.
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    def __call__(self, img):
        return self.base_transform(img)


ILSVRC_class_index = json.load(open('./data/imagenet_class_index.json', 'r'))
ILSVRC_class_index


# 출력 결과에서 라벨을 예측하는 후처리 클래스 

class ILSVRCPredictor():
    
    def __init__(self, class_index):
        self.class_index = class_index 
        
    def predict_max(self, out):
        maxid = torch.argmax(out, 1).item()
        # 가장 예측 확률이 높은 라벨명.
        predicted_label_name = self.class_index[str(maxid)][1]
        
        return predicted_label_name


ILSVRC_class_index['207']


predictor = ILSVRCPredictor(ILSVRC_class_index)

# input lmage load
image_file_path = './data/goldenretriever-3724972_960_720.jpg'
img = Image.open(image_file_path)

resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transform = BaseTransform(resize, mean, std)
img_transformed = transform(img)

# batch_size = 1임을 보여주려고,
inputs = img_transformed.unsqueeze_(0)
