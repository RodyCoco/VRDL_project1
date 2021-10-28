import os
from PIL import Image
import natsort
import torch
import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms as tfs
import numpy as np
GPU_NUMBER = 0

def load_class(path = "2021VRDL_HW1_datasets/classes.txt"):
    with open(path, newline='') as fh:
        cls = fh.readlines()
        for i in range(len(cls)):
            cls[i] = cls[i].strip("\n")
    return cls

def load_train_label(path = "2021VRDL_HW1_datasets/training_labels.txt"):
    with open(path, newline='') as fh:
        data = fh.readlines()
        L = []
        for item in data:
            item=item.strip('\n\r')
            temp = item.split(" ")
            L.append([temp[0],temp[1]])
        L.sort(key = lambda s:s[0])
        for i in  range(len(L)):
            # L[i] = one_hot_vector(int(L[i][1][0:3]))
            L[i] = int(L[i][1][0:3])-1
    return torch.tensor(L)

default_trans = tfs.Compose([
    tfs.Resize((224, 224), Image.BILINEAR),
    tfs.ToTensor(),
    tfs.Normalize(mean=[0.5,0.5,0.5], std=[0.2, 0.2, 0.2])])

class TrainDataSet():
    def __init__(self, main_dir="2021VRDL_HW1_datasets/training_images", transform = default_trans):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)
        self.labels= load_train_label()

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs [idx])
        image = Image.open(img_loc).convert("RGB")
        image = self.transform(image)
        # tensor_image = tensor_image.unsqueeze(0)
        return image

def get_dataset(transform= default_trans):
    L=[]
    temp = TrainDataSet(transform=transform)
    for idx,img in enumerate(temp):
        L.append(img)
    
    L=torch.tensor([item.cpu().detach().numpy() for item in L])
    return L