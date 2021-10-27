import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import numpy as np
from PIL import Image 
import torchvision.models as models
from IPython.display import display
from data_gen import CustomDataSet, get_dataset, load_class, load_eval_label, load_train_label, GPU_NUMBER,get_origin_dataset
from model import ResNet, ResNet50, model_urls
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.model_zoo import load_url as load_state_dict_from_url

lr = 0.001
epochs = 60
batch_size = 16

def procedure():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)

    bird_class = load_class()
    origin_train_label = load_eval_label()
    train_label = load_train_label(origin_train_label)

    
    model = models.resnet50(pretrained=True).cuda(GPU_NUMBER)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 200).cuda(GPU_NUMBER)
    model.double()
    print("model done")

    train_data = get_dataset("2021VRDL_HW1_datasets/training_images")
    origin_train_data = get_origin_dataset("2021VRDL_HW1_datasets/training_images")
    train_dataset = Data.TensorDataset(train_data, train_label)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    origin_train_dataset = Data.TensorDataset(origin_train_data, origin_train_label)
    origin_train_loader = DataLoader(dataset=origin_train_dataset, batch_size=batch_size, shuffle=True)
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

    # train_dataset = datasets.DatasetFolder('2021VRDL_HW1_datasets/training_images', transform=transform)
    # test_dataset = datasets.DatasetFolder('2021VRDL_HW1_datasets/testing_images', transform=transform)

    for epoch in range(epochs) :
        train_loss = train(model, train_loader, loss_function, optimizer)
        
        print(epoch+1,": ",eval_acc(model,origin_train_loader)," ",train_loss)
        scheduler.step()
        writer.add_scalar("Loss/train_loss", train_loss, epoch + 1)
        torch.save(model.state_dict(), f'/tmp/resnet50_{epoch+1}.pkl')
        print("Save model")
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
        out = model(x.type(torch.DoubleTensor).cuda(GPU_NUMBER)).double().cuda(GPU_NUMBER)
        optimizer.zero_grad()
        # print(out.shape,y.shape)
        loss = loss_function(out,y)
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
            tmp = []
            for idx, item in enumerate(out):
                tmp.append(np.argmax(item.cpu().detach().numpy()))
            tmp = np.asarray(tmp)
            y = np.asarray(y.cpu().detach().numpy())
            for item in (y-tmp):
                if item == 0:
                    acc +=1
    return (acc/number_of_data)

if __name__ == '__main__':
    procedure()