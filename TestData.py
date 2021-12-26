import os
import pandas as pd
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset
import torch
from torchvision import transforms
import cv2
global train_root
from PIL import Image

train_root = r"2021-intro-to-ml-and-dl-final-project\test\test"

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

    def __len__(self):
        return (len(self.img))

    def __getitem__(self, idx):
        Img_pth = self.img[idx]
        file_path = os.path.join(train_root, Img_pth)
        try:Img = cv2.imread(file_path)
        except: 
            Img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2BGR)
            Img = Img.transpose(2,0,1)
        _Img = transform(Image.fromarray(Img))


        return _Img