import os
import pandas as pd
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset
import torch
from torchvision import transforms
from scipy.ndimage.filters import gaussian_filter
import cv2
global train_root,label_root
from PIL import Image

train_root = r"2021-intro-to-ml-and-dl-final-project/train/train"
label_root = r"2021-intro-to-ml-and-dl-final-project/train_data.csv"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])
class ImageDataset(Dataset):
    def __init__(self):
        self.img = list(os.listdir(train_root))
        ori_len = len(self.img)
        for m in range(ori_len):
            self.img.append("extraG_{:.0f}".format(m))
        for n in range(ori_len):
            self.img.append("extraC_{:.0f}".format(n))
        for o in range(ori_len):
            self.img.append("extraD_{:.0f}".format(o))
        self.label_pth = label_root

    def __len__(self):
        return (len(self.img))

    def __getitem__(self, idx):
        Img_pth = self.img[idx]
        if Img_pth[0:6] == "extraG":
            Img_pth = self.img[int(Img_pth[7:])]
            file_path = os.path.join(train_root, Img_pth)
            try:Img = cv2.imread(file_path)
            except: 
                Img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2BGR)
                Img = Img.transpose(2,0,1)
            Img = gaussian_filter(Img,2.5)
            _Img = transform(Image.fromarray(Img))


            labelname = pd.read_csv(label_root)["Name"].to_numpy()
            index = np.where(labelname == Img_pth)[0]
            label = pd.read_csv(label_root)["Type"].to_numpy()[index]
            return _Img,label[0]
        elif Img_pth[0:6] == "extraC":
            Img_pth = self.img[int(Img_pth[7:])]
            file_path = os.path.join(train_root, Img_pth)
            try:Img = cv2.imread(file_path)
            except: 
                Img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2BGR)
                Img = Img.transpose(2,0,1)
            _Img = transform(Image.fromarray(Img))
            tf = transforms.ColorJitter(0.5,0.5,0.5,0.5)
            _Img = tf(_Img)


            labelname = pd.read_csv(label_root)["Name"].to_numpy()
            index = np.where(labelname == Img_pth)[0]
            label = pd.read_csv(label_root)["Type"].to_numpy()[index]
            return _Img,label[0]
        elif Img_pth[0:6] == "extraD":
            Img_pth = self.img[int(Img_pth[7:])]
            file_path = os.path.join(train_root, Img_pth)
            try:Img = cv2.imread(file_path)
            except: 
                Img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2BGR)
                Img = Img.transpose(2,0,1)
            _Img = transform(Image.fromarray(Img))
            tf = transforms.RandomRotation(90)
            _Img = tf(_Img)


            labelname = pd.read_csv(label_root)["Name"].to_numpy()
            index = np.where(labelname == Img_pth)[0]
            label = pd.read_csv(label_root)["Type"].to_numpy()[index]
            return _Img,label[0]
        else:
            file_path = os.path.join(train_root, Img_pth)
            try:Img = cv2.imread(file_path)
            except: 
                Img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2BGR)
                Img = Img.transpose(2,0,1)
            _Img = transform(Image.fromarray(Img))


            labelname = pd.read_csv(label_root)["Name"].to_numpy()
            index = np.where(labelname == self.img[idx])[0]
            label = pd.read_csv(label_root)["Type"].to_numpy()[index]
            return _Img,label[0]