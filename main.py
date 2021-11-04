import torch
import torchvision.transforms as tfs
import numpy as np
from torch.utils.data import DataLoader
from model import Vit_large_patch16_224
from PIL import Image
from data_gen import BirdDataSet, load_label, GPU_NUMBER

lr = 5e-6
epochs = 25
batch_size = 32
number_of_data_per_class = 15

train_trans = tfs.Compose([
    tfs.Resize((224, 224), Image.BILINEAR),
    tfs.RandomHorizontalFlip(),
    tfs.RandomRotation(degrees=45),
    tfs.ToTensor(),
    tfs.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
])

valid_trans = tfs.Compose([
    tfs.Resize((224, 224), Image.BILINEAR),
    tfs.ToTensor(),
    tfs.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
])

# indicate images that chosen in validation set
valid_index = [[0, 2], [3, 5], [6, 8], [9, 11], [12, 14]]


def procedure(id):

    print(f"round: {id+1}")

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)

    model_name = "Vit_large_patch16_224"
    model = Vit_large_patch16_224().cuda(GPU_NUMBER)
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model.double()
    print(f"load {model_name} done")

    train_data = []
    valid_data = []
    label = load_label()
    for i in range(len(label)):
        if i % number_of_data_per_class >= valid_index[id][0] \
                and i % number_of_data_per_class <= valid_index[id][1]:
            valid_data.append(label[i])
        else:
            train_data.append(label[i])

    train_data = BirdDataSet(
        main_dir="2021VRDL_HW1_datasets/training_images",
        data=train_data,
        transform=train_trans)
    train_loader = DataLoader(
        dataset=train_data, batch_size=batch_size, num_workers=4, shuffle=True)
    valid_data = BirdDataSet(
        main_dir="2021VRDL_HW1_datasets/training_images",
        data=valid_data, transform=valid_trans)
    valid_loader = DataLoader(
        dataset=valid_data, batch_size=batch_size, num_workers=4, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: 0.99 ** (epoch))
    min_loss = np.inf

    for epoch in range(epochs):
        train_loss = train(model, train_loader, loss_function, optimizer)
        valid_acc = eval_acc(model, valid_loader, len(valid_data))
        valid_loss = eval_loss(model, valid_loader, loss_function)
        print(
            "Epoch[{:3d}/{}]  train_loss:{} valid_acc:{} valid_loss:{}"
            .format(epoch+1, epochs, train_loss, valid_acc, valid_loss)
            )
        scheduler.step()
        if valid_loss < min_loss:
            torch.save(model.state_dict(), f"{model_name}_{id}.pkl")
            min_loss = valid_loss
            print(f"Save {model_name}_{id}.pkl")


def train(model, train_loader, loss_function, optimizer):

    model.train()
    loss_list = []

    for index, (x, y) in enumerate(train_loader):

        out = model(
            x.type(torch.DoubleTensor)
            .cuda(GPU_NUMBER)).double().cuda(GPU_NUMBER)
        optimizer.zero_grad()
        loss = loss_function(out, y.cuda(GPU_NUMBER))
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()

    return sum(loss_list) / len(loss_list)


def eval_acc(model, loader, data_len):
    model.eval()
    acc = 0
    with torch.no_grad():
        for index, (x, y) in enumerate(loader):
            out = model(x.type(torch.DoubleTensor).cuda(GPU_NUMBER)).double()
            tmp = np.asarray(
                [np.argmax(item.cpu().detach().numpy()) for item in out])
            y = np.asarray(y.cpu().detach().numpy())
            acc += np.count_nonzero((y-tmp) == 0)
    return (acc/data_len)


def eval_loss(model, loader, loss_function):
    model.eval()
    loss_list = []
    with torch.no_grad():
        for index, (x, y) in enumerate(loader):
            out = model(x.type(torch.DoubleTensor).cuda(GPU_NUMBER)).double()
            loss = loss_function(out, y.cuda(GPU_NUMBER))
            loss_list.append(loss.item())
    return sum(loss_list) / len(loss_list)

if __name__ == '__main__':
    # use 5-fold cross-validation
    for i in range(5):
        procedure(i)
