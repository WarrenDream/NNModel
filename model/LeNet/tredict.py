import numpy as np
np.random.seed(42)

from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
import argparse
import imutils
import torch
import cv2 

ap=argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str,required=True,help="path to the trained PyTorch model")
args = vars(ap.parse_args())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] loading the KMNIST test dataset ...")
testData=KMNIST(root="data", train=False, download=True, transform=ToTensor())
idxs=np.random.choice(range(0,len(testData)),size=(10,))
testData = Subset(testData, idxs)

testDataLoader = DataLoader(testData,batch_size=1)

model=torch.load(args["model"]).to(device)
model.eval()

with torch.no_grad()
    for(image, label) in testDataLoader:
        origImage = image.numpy().squeeze(axis=(0,1))
        gtLabel=testData.dataset.classes[label.numpy()[0]]

        image=image.to(device)
        pred=model(image)

        idx=pred.argmax(axis=1).cpu().numpy()[0]
        predLabel=testData.dataset.classes[idx]

        origImage=np.dstack([origImage]*3)
        origImage=imutils.resize(origImage,width=128)

        color=(0,255,0)if gtLabel == predLabel else (0, 0, 255)
        cv2.putText(origImage,gtLabel,(2,255),cv2.FONT_HERSHEY_SIMPLEX,0.95,color,2)

        print("[INFO] ground truth label: {}, predicted label:{}".format(gtLabel,predLabel))
        cv2.imshow("image",origImage)
        cv2.waitKey(0)


#cmd use
#python predict.py --model output/model.pth
