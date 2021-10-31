import os
from PIL import Image
import natsort
import torch
import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms as tfs
import numpy as np
import torch.utils.data as data

GPU_NUMBER = 0

def load_class(path = "2021VRDL_HW1_datasets/classes.txt"):
    with open(path, newline='') as fh:
        cls = fh.readlines()
        for i in range(len(cls)):
            cls[i] = cls[i].strip("\n")
    return cls

def load_label(path = "2021VRDL_HW1_datasets/training_labels.txt"):
    with open(path, newline='') as fh:
        data = fh.readlines()
        L = []
        for item in data:
            item=item.strip('\n\r')
            temp = item.split(" ")
            L.append([temp[0],temp[1]])
        L.sort(key = lambda s:s[1])
        for i in  range(len(L)):
            # L[i] = one_hot_vector(int(L[i][1][0:3]))
            L[i][1] = int(L[i][1][0:3])-1
    return L

class BirdDataSet(data.Dataset):
    def __init__(self, data, main_dir, transform):
        super(BirdDataSet, self).__init__()
        self.main_dir = main_dir
        self.transform = transform
        self.data =  data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.data[idx][0])
        image = Image.open(img_loc).convert("RGB")
        image = self.transform(image)
        # tensor_image = tensor_image.unsqueeze(0)
        return image,  self.data[idx][1]