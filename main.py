import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import numpy as np
from PIL import Image 
from IPython.display import display
from data_gen import CustomDataSet, get_dataset, load_class, load_train_label
from model import ResNet, ResNet50, softmax
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

lr = 0.001
epochs = 40
batch_size = 20

def procedure():

    bird_class = load_class()
    train_label = load_train_label()

    model = ResNet50()
    model.double()
    train_data = get_dataset("2021VRDL_HW1_datasets/training_images")

    train_dataset = Data.TensorDataset(train_data, train_label)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # print(train_data[0].shape)
    # plt.imshow(train_dataset[0].swapaxes(0,1).swapaxes(1,2))
    # plt.show()

    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    writer = SummaryWriter()
    loss_function = torch.nn.CrossEntropyLoss()
    decay = lambda epoch: 0.99 ** (epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=decay)
    min_loss = np.inf

    # train_dataset = datasets.DatasetFolder('2021VRDL_HW1_datasets/training_images', transform=transform)
    # test_dataset = datasets.DatasetFolder('2021VRDL_HW1_datasets/testing_images', transform=transform)

    for epoch in range(epochs) :
        print(len(train_loader))
        train_loss = train(model, train_loader, loss_function, optimizer)
        
        print(eval_acc(model,train_loader))
        scheduler.step()

        writer.add_scalar("Loss/train_loss", train_loss, epoch + 1)
        # writer.add_scalar("Loss/val_loss", valid_loss, epoch + 1)
        # writer.add_scalar("Loss/test_loss", test_loss, epoch + 1)
        # writer.add_scalars("Loss/total_loss", {'train_loss':train_loss, 'val_loss':valid_loss, 'test_loss':test_loss}, epoch + 1)
        # writer.flush()

        # print("Epoch[{:3d}/{}]\ttrain:{:5f}, valid:{:5f}, test:{:5f}".format(epoch+1, epochs, train_loss, valid_loss, test_loss))
        # if valid_loss < min_loss:
        #     torch.save(model.state_dict(), 'model.pkl')
        #     print("Save model")
        #     min_loss = valid_loss
            
    writer.close()

def train(model, train_loader, loss_function, optimizer):
    model.train()
    loss_list = []

    for index, (x,y) in enumerate(train_loader):
        print(index)
        out = model(x.type(torch.DoubleTensor)).double()
        optimizer.zero_grad()
        # print(out.shape,y.shape)
        loss = loss_function(out,y)
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()

    return sum(loss_list) / len(loss_list)

def eval_acc(model,loader):
    model.eval()
    acc = 0
    L = len(loader)
    for index, (x,y) in enumerate(loader):
        for idx, item in enumerate(x):
            predict_label = np.argmax(item)+1
            if predict_label == y[idx]:
                acc += 1

    return (acc/L)

if __name__ == '__main__':
    procedure()