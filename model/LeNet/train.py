#设置图片后端保存
import matplotlib
matplotlib.use("Agg")
#导入带训练网络
from lenet import LeNet
#导入依赖包
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt 
import numpy as np 
import argparse
import torch
import time

ap =argparse.ArgumentParser()
ap.add_argument("-m","--model",type=str,required=True,help="path to output trained model")
ap.add_argument("-p","--plot",type=str,required=True,help="path to output loss/accuracy plot")
args=vars(ap.parse_args())

INIT_LR=1e-3
BATCH_SIZE=64
EPOCHS=10

TRAIN_SPLIT=0.75
VAL_SPLIT=1-TRAIN_SPLIT

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] loading the KMNIST dataset ...")
trainData=KMNIST(root="data",train=True,download=True,transform=ToTensor())
testData=KMNIST(root="data",train=False,download=True,transform=ToTensor())

print("[INFO] generating the train/validation split ...")
numTrainSamples=int(len(trainData)*TRAIN_SPLIT)
numValSamples=int(len(trainData)*VAL_SPLIT)
(trainData,valData)=random_split(trainData,[numTrainSamples,numValSamples],generator=torch.Generator().manual_seed(42))

trainDataLoader=DataLoader(trainData,shuffle=True,batch_size=BATCH_SIZE)
valDataLoader=DataLoader(valData,batch_size=BATCH_SIZE)
testDataLoader=DataLoader(testData,batch_size=BATCH_SIZE)

trainSteps = len(trainDataLoader.dataset)//BATCH_SIZE
valSteps = len(valDataLoader.dataset)//BATCH_SIZE

print("[INFO] initializing the LeNet model ...")
model = LeNet(numChannels=1,classes=len(trainData.dataset.classes)).to(device)

opt = Adam(model.parameters(),lr=INIT_LR)
lossFn=nn.NLLLOSS()

H={
    "train_loss":[],
    "train_acc":[],
    "val_loss":[],
    "val_acc":[]
}

print("[INFO] training the network ...")
startTime = time.time()

for e in range(0,EPOCHS):
    model.train()

    totalTrainLoss=0
    totalValLoss=0

    trainCorrect=0
    valCorrect=0

    for(x,y) in trainDataLoader:
        (x,y)=(x.to(device),y.to(device))

        pred=model(x)
        loss=lossFn(pred,y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        totalTrainLoss +=loss
        trainCorrect +=(pred.argmax(1)==y).type(torch.float).sum().item()

    with torch.no_grad():
        model.eval()
        
        for (x,y) in valDataLoader:
            (x,y)=(x.to(device),y.to(device))

            pred = model(x)
            totalValLoss+=lossFn(pred,y)

            valCorrect +=(pred.argmax(1)==y).type(torch.float).sum().item()

    avgTrainLoss = totalTrainLoss/trainSteps
    avgValLoss = totalValLoss/valSteps
    
    trainCorrect = trainCorrect/len(trainDataLoader.dataset)
    valCorrect=valCorrect/len(valDataLoader.dataset)

    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["train_acc"].append(trainCorrect)
    H["val_loss"].append(avgValLoss.cpu().detach().numpy())
    H["val_acc"].append(valCorrect)

    print("[INFO] EPOCH: {}/{}".format(e+1,EPOCHS))
    print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(avgTrainLoss,trainCorrect))
    print("Val loss: {:.6f}, Val accuracy: {:.4f}".format(avgValLoss,valCorrect))

endTime=time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(endTime-startTime))

print("[INFO] evaluation network ...")

witch torch.no_grad():
    model.eval()
    preds=[]
    for (x,y) in testDataLoader:
        x=x.to(device)

        pred=model(x)
        preds.extend(pred.argmax(axis=1).cpu().numpy())
print(classification_report(testData.targets.cpu().numpy(),np.array(preds),target_names=testData.classes))

plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"],label="train_loss")
plt.plot(H["val_loss"],label="val_loss")
plt.plot(H["train_acc"],label="train_acc")
plt.plot(H["val_acc"],label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

torch.save(model,args["model"])

#cmd to use 
#python train.py --model output/model.pth --plot output/plot.png



