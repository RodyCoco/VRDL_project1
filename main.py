import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import numpy as np
from PIL import Image 
import torchvision.models as models
from IPython.display import display
from data_gen import CustomDataSet, get_dataset, load_train_label, GPU_NUMBER
from model import ResNet, ResNet50
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime

lr = 0.001
epochs = 60
batch_size = 32

def procedure():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    
    model = models.resnet152(pretrained=True).cuda(GPU_NUMBER)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 200).cuda(GPU_NUMBER)
    model = torch.nn.DataParallel(model, device_ids=[1, 2, 3])
    model.double()
    print("model done")

    # bird_class = load_class()
    train_label = load_train_label()
    train_data = get_dataset("2021VRDL_HW1_datasets/training_images")
    train_dataset = Data.TensorDataset(train_data, train_label)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    print("train_data  done")

    # for i in range(0,11):
    #     plt.imshow(train_data[0][i].cpu().permute(1, 2, 0))
    #     plt.savefig(f"test{i}.png")
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    writer = SummaryWriter()
    loss_function = torch.nn.CrossEntropyLoss()
    decay = lambda epoch: 0.99 ** (epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=decay)
    min_loss = np.inf

    for epoch in range(epochs) :
        print("train:",datetime.datetime.now())
        train_loss = train(model, train_loader, loss_function, optimizer)
        print("after train:",datetime.datetime.now())
        print(epoch+1,": ",eval_acc(model,train_loader)," ",train_loss)
        scheduler.step()
        writer.add_scalar("Loss/train_loss", train_loss, epoch + 1)
        torch.save(model.state_dict(), f'/tmp/resnet50_{epoch+1}.pkl')
        print("Save model")
   

def train(model, train_loader, loss_function, optimizer):
    model.train()
    loss_list = []

    for index, (x,y) in enumerate(train_loader):
        out = model(x.type(torch.DoubleTensor).cuda(GPU_NUMBER)).double().cuda(GPU_NUMBER)
        optimizer.zero_grad()
        # print(out.shape,y.shape)
        loss = loss_function(out,y.cuda(GPU_NUMBER))
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()

    return sum(loss_list) / len(loss_list)

def eval_acc(model,loader,number_of_data = 3000):
    model.eval()
    acc = 0
    with torch.no_grad():
        for index, (x,y) in enumerate(loader):
            out = model(x.type(torch.DoubleTensor).cuda(GPU_NUMBER)).double()
            tmp = np.asarray([np.argmax(item.cpu().detach().numpy()) for item in out])
            y = np.asarray(y.cpu().detach().numpy())
            acc += np.count_nonzero((y-tmp) == 0)
    return (acc/number_of_data)

if __name__ == '__main__':
    procedure()