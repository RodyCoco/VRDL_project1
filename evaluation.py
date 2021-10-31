from model import Densenet161
from data_gen import load_class, GPU_NUMBER, BirdDataSet
import torchvision.models as models
import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
import torchvision.transforms as tfs
import matplotlib.pyplot as plt

batch_size = 16

test_trans = tfs.Compose([
    tfs.Resize((224, 224), Image.BILINEAR),
    tfs.ToTensor()
])

def procedure():
    bird_class = load_class()
    model = Densenet161().cuda(GPU_NUMBER)
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    name = "densenet161.pkl"
    model.load_state_dict(torch.load(name))
    print(name)
    model.double()
    model.eval()
    print("load model done")

    
    with open('2021VRDL_HW1_datasets/testing_img_order.txt') as f:
        test_images = [[x.strip(),0] for x in f.readlines()]  # all the testing images
    submission = []

    test_data = BirdDataSet(main_dir="2021VRDL_HW1_datasets/testing_images"\
        ,data = test_images, transform = test_trans)
    test_loader = DataLoader(dataset=test_data, batch_size=1, num_workers=4)

    with torch.no_grad():
        for idx, (img,_) in enumerate(test_loader):
            out = model(img.type(torch.DoubleTensor).cuda(GPU_NUMBER)).double()
            predicted_class = np.asarray([np.argmax(item.cpu().detach().numpy()) for item in out])[0]
            print(idx)
            submission.append([test_images[idx][0], bird_class[predicted_class]])
    np.savetxt('answer.txt', submission, fmt='%s')
    
if __name__ == '__main__':
    procedure()