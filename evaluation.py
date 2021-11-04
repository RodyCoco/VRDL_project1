import torch
import numpy as np
import torchvision.transforms as tfs
from torch.utils.data import DataLoader
from PIL import Image
from model import Vit_large_patch16_224
from data_gen import load_class, GPU_NUMBER, BirdDataSet

batch_size = 1
num_model = 5

test_trans = tfs.Compose([
    tfs.Resize((224, 224), Image.BILINEAR),
    tfs.ToTensor(),
    tfs.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
])


def procedure():
    bird_class = load_class()
    models = []
    for i in range(num_model):
        model = Vit_large_patch16_224().cuda(GPU_NUMBER)
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
        name = f"Vit_large_patch16_224_{i}.pkl"
        model.load_state_dict(torch.load(name))
        model.double()
        model.eval()
        models.append(model)

    print("load model done")

    with open('2021VRDL_HW1_datasets/testing_img_order.txt') as f:
        test_images = \
            [[x.strip(), 0] for x in f.readlines()]  # all the testing images
    submission = []

    test_data = BirdDataSet(main_dir="2021VRDL_HW1_datasets/testing_images",
                            data=test_images, transform=test_trans)
    test_loader = DataLoader(dataset=test_data, batch_size=1, num_workers=4)

    with torch.no_grad():
        for idx, (img, _) in enumerate(test_loader):
            print(idx)
            sum = torch.zeros((1, 200)).cuda(GPU_NUMBER).double()
            for i in range(num_model):
                out = models[i](
                    img.type(torch.DoubleTensor).cuda(GPU_NUMBER)).double()
                out = torch.nn.Softmax(dim=1)(out)
                sum += out
            predicted_class = np.asarray(
                [np.argmax(item.cpu().detach().numpy())for item in sum])[0]
            submission.append(
                [test_images[idx][0], bird_class[predicted_class]])
            print([test_images[idx][0], bird_class[predicted_class]])
    np.savetxt('answer.txt', submission, fmt='%s')

if __name__ == '__main__':
    procedure()
