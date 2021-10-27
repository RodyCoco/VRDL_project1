from data_gen import CustomDataSet, get_dataset, load_class, load_train_label, GPU_NUMBER
import torchvision.models as models
import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from torchvision import datasets, transforms

batch_size = 16

def procedure():
    bird_class = load_class()

    model = models.resnet50().cuda(GPU_NUMBER)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 200).cuda(GPU_NUMBER)
    model.load_state_dict(torch.load("resnet50_29.pkl"))
    model.double()
    model.eval()
    print("load model done")

    # test_data = get_dataset("2021VRDL_HW1_datasets/testing_images")
    # test_dataset = Data.TensorDataset(test_data)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    # tmp = []
    # with torch.no_grad():
    #     for index, x in enumerate(test_loader):
    #         out = model(x.type(torch.DoubleTensor).cuda(GPU_NUMBER)).double()
    #         for idx, item in enumerate(out):
    #             tmp.append(np.argmax(item.cpu().detach().numpy())+1)
    # tmp = np.asarray(tmp)
    with open('2021VRDL_HW1_datasets/testing_img_order.txt') as f:
        test_images = [x.strip() for x in f.readlines()]  # all the testing images
    submission = []
    print(len(test_images))
    with torch.no_grad():
        for img_name in test_images:  # image order is important to your result
            transform = transforms.Compose([transforms.Resize(255),transforms.CenterCrop(224)])
            image = Image.open("2021VRDL_HW1_datasets/testing_images/"+img_name).convert("RGB")
            tensor_image = transform(image)
            tensor_image = tensor_image.unsqueeze(0)
            tensor_image = torch.tensor(tensor_image).clone().detach().cuda(GPU_NUMBER)
            out = model(tensor_image.double())  # the predicted category
            predicted_class = np.argmax(out.cpu().detach().numpy())
            submission.append([img_name, bird_class[predicted_class]])
    np.savetxt('answer.txt', submission, fmt='%s')
    print("done")
if __name__ == '__main__':
    procedure()