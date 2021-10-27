import os
from PIL import Image
import natsort
import torch
import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms as tfs


GPU_NUMBER = 0
num_augmetation = 5
mean = [0.5, 0.5, 0.5]
std = [0.1, 0.1, 0.1]

def load_class(path = "2021VRDL_HW1_datasets/classes.txt"):
    with open(path, newline='') as fh:
        cls = fh.readlines()
        for i in range(len(cls)):
            cls[i] = cls[i].strip("\n")
    return cls

def one_hot_vector(n,dim=200):
    L=[0]*dim
    L[n-1] = 1
    return L

def load_eval_label(path = "2021VRDL_HW1_datasets/training_labels.txt"):
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
        return torch.tensor(L).cuda(GPU_NUMBER)

def load_train_label(data):
    L = []
    for item in data:
        for i in range(num_augmetation):
            L.append(item)
    return torch.tensor(L).cuda(GPU_NUMBER)

class CustomDataSet():
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs [idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        # tensor_image = tensor_image.unsqueeze(0)
        return tensor_image

def get_origin_dataset(path):
    L=[]
    transform = tfs.Compose([tfs.Resize(255),\
        tfs.CenterCrop(224),tfs.ToTensor(),tfs.Normalize(mean, std)])
    temp = CustomDataSet(path,transform=transform)
    for index in range(len(temp)):
        L.append(temp[index])
    L=torch.tensor([item.cpu().detach().numpy() for item in L]).cuda(GPU_NUMBER)
    return L

def get_dataset(path):
    L=[]
    transform = tfs.Compose([tfs.Resize(255),tfs.CenterCrop(224),tfs.ToTensor()])
    temp = CustomDataSet(path,transform=transform)
    f = tfs.Compose([tfs.ToPILImage(),tfs.RandomHorizontalFlip(p=0.5),\
        tfs.RandomResizedCrop((224,224)),tfs.RandomRotation([0,45],resample=Image.BICUBIC),tfs.ColorJitter(brightness=0.5, hue=0.3),tfs.ToTensor()\
            ,tfs.Normalize(mean, std)])
    for idx,img in enumerate(temp):
        print(idx)
        L.append(img)
        for i in range(num_augmetation-1):
            tmp_img = f(img)
            L.append((tmp_img))
    L=torch.tensor([item.cpu().detach().numpy() for item in L]).cuda(GPU_NUMBER)
    return L