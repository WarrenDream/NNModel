#设置图片后端保存
import matplotlib
matplotlib.use("Agg")
#导入带训练网络
from resnet import resnet18
#导入依赖包
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
import torch.utils.data as Data
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
from torchvision.datasets import FashionMNIST

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import argparse
import time
import copy

#处理训练数据集
def train_data_process():
    train_data=FashionMNIST(
        root="./data/FashionMNIST",
        train=True,
        transform=transforms.Compose([transforms.Resize(size=224),transforms.ToTensor()]),
        download=True,
    )
    train_loader=Data.DataLoader(
        datase=train_data,
        batch_size=64,
        shuffle=False,
        num_workers=2,
    )
    print("the numeber of batch in train_loader:",len(train_loader))
    for step, (b_x, b_y) in enumerate(train_loader):
        if step>0:
            break
    batch_x=b_x.squeeze().numpy()  # 将四维张量移除第1维，并转换成Numpy数组
    batch_y=b_y.numpy() # 将张量转换成Numpy数组
    class_label=train_data.classes  # 训练集的标签
    class_label[0]="T-shirt"
    print("the size of batch in train data:",batch_x.shape)

    plt.figure(figsize=(12,5))
    for ii in np.arange(len(batch_y)):
        plt.subplot(4,16,ii+1)
        plt.imshow(batch_x[ii,:,:], cmap=plt.cm.gray)
        plt.title(class_label[batch_y[ii]],size=9)
        plt.axis("off")
        plt.subplots_adjust(wspace=0.05)
    plt.show()
    return train_loader, class_label
# 处理测试集数据
def test_data_process():
    test_data=FashionMNIST(
        root="./data/FashionMNIST",
        train=False,
        transform=transforms.Compose(
        [transforms.Resize(size=224),transforms.ToTensor()]
        ),
        download=True,
       )
    test_loader=Data.DataLoader(
        dataset=test_data,
        batch_size=1,
        shuffle=True,
        num_workers=2,
    )
    for step,(b_x,b_y) in enumerate(test_loader):
        if step >0:
            break
    batch_x = b_x.squeeze().numpy()
    batch_y = b_y.numpy()
    print("the size of batch in test data:", batch_x.shape)
    return test_loader
#定义网络的训练过程
def train_model(model, traindataloader, train_rate, criterion, device, optimizer, num_epochs=100):
    """
    :param model: 网络模型
    :param traindataloader: 训练数据集，会切分为训练集和验证集
    :param train_rate: 训练集batch_size的百分比
    :param criterion: 损失函数
    :param device: 运行设备
    :param optimizer: 优化方法
    :param num_epochs: 训练的轮数
    """
    batch_num=len(traindataloader)
    train_batch_num=round(batch_num*train_rate)
    best_model_wts=copy.deepcopy(model.state_dict())
    #初始化参数
    best_acc = 0.0 
    train_loss_all=[]
    train_acc_all=[]
    val_loss_all=[]
    val_acc_all=[]
    since = time.time()
    #进行迭代训练模型
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs-1))
        print("-"*10)
        #初始化参数
        train_loss =0.0
        train_corrects=0
        train_num=0
        val_loss=0.0
        val_corrects=0
        val_num=0
        #mini batch 训练和计算
        for step , (b_x,b_y) in enumerate(traindataloader):
            b_x=b_x.to(device)
            b_y=b_y.to(device)
            if step < train_batch_num:
                model.train()
                output=model(b_x)
                pre_lab=torch.argmax(output,1)
                loss=criterion(output, b_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss+=loss.item()*b_x.size(0)
                train_corrests+=torch.sum(pre_lab==b_y.data)
                train_num +=b_x.size(0)
            else:
                model.eval()
                output=model(b_x)
                pre_lab=torch.argmax(output,1)
                loss=criterion(output,b_y)
                val_loss+=loss.item()*b_x.size(0)
                val_corrects+=torch.sum(pre_lab==b_y.data)
                val_num +=b_x.size(0)
        train_loss_all.append(train_loss/train_num)
        train_acc_all.append(train_corrects.double().item()/train_num)
        val_loss_all.append(val_loss/val_num)
        val_acc_all.append(val_corrects.double().item()/val_num)
        print("{} train loss: {:.4f} train acc: {:.4f}".format(epoch,train_loss_all[-1],train_acc_all[-1]))
        print("{} val loss: {:.4f} val acc: {:.4f}".format(epoch,val_loss_all[-1],val_acc_all[-1]))
        
        if val_acc_all[-1]>best_acc:
            best_acc=val_acc_all[-1]
            best_model_wts=copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, "./weights/best_weight.pth")
        time_use = time.time()-since
        print("train and val complete in {:.0f}m {:.0f}s".format(time_use//60, time_use%60))
        
        if epoch %10==0:
            torch.save(model.state_dict(),"./weights/epoch{}_weight.pth".format(epoch))
    #选择最优参数
    model.load_state_dict(best_model_wts)
    train_process = pd.DataFrame(
        data={
            "epoch":range(num_epochs),
            "train_loss_all": train_loss_all,
            "val_loss_all": val_loss_all,
            "train_acc_all": train_acc_all,
            "val_acc_all": val_acc_all,
        }
    )

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_process["epoch"], train_process.train_loss_all,"ro-",label="train loss")
    plt.plot(train_process["epoch"],train_process.val_loss_all,"bs-", label="val_loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.subplot(1,2,2)
    plt.plot(train_process["epoch"],train_process.train_acc_all,"ro-",label="train_acc")
    plt.plot(train_process["epoch"],train_process.val_acc_all, "bs-", label="val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()
    return model, train_process

# 测试模型
def test_model(model, testdataloader, device):
    """
    :param model: 网络模型
    :param testdataloader: 测试数据集
    :param device: 运行设备
    """
    #初始化参数
    test_corrects=0.0
    test_num=0
    test_acc=0.0

    with torch.no_grad():
        for test_data_x, test_data_y in testdataloader:
            test_data_x=test_data_x.to(device)
            test_data_y=test_data_y.to(device)
            model.eval()
            output=model(test_data_x)
            pre_lab=torch.argmax(output, 1)
            test_corrects+=torch.sum(pre_lab==test_data_y.data)
            test_num+=test_data_x.size(0)
    test_acc=test_corrects.double().item()/test_num
    print("test accuracy:",test_acc)

#模型训练和测试
def train_model_process(myconvnet):
    optimizer= torch.optim.Adam(myconvnet.parameters(),lr=0.001)
    criterion=nn.CrossEntropyLoss()
    device="cuda" if torch.cuda.is_available() else "cpu"
    train_loader, class_label =train_data_process()
    test_loader = test_data_process()

    myconvnet = myconvnet.to(device())
    myconvnet, train_process=train_model(
        myconvnet,train_loader, 0.8,criterion, device, optimizer,num_epochs=100
    )
    test_model(myconvnet, test_loader, device)


if __name__=="__main__":
    model=resnet18()
    train_model_process(model)
ap =argparse.ArgumentParser()
ap.add_argument("-m","--model",type=str,required=True,help="path to output trained model")
ap.add_argument("-p","--plot",type=str,required=True,help="path to output loss/accuracy plot")
args=vars(ap.parse_args())




#cmd to use 
#python train.py --model output/model.pth --plot output/plot.png



