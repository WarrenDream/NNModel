import torch
import torch.nn as nn
from torch.nn import BatchNorm2d
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch.nn import AvgPool2d
from torch.utils import model_zoo
from torch import flatten

__all__ = ["ResNet", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
#浅层网络（18、34）基础模块
class BasicBlock(Module):
    expansion=1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock,self).__init__()
        self.conv1=conv3x3(inplanes,planes,stride)
        self.bn1=BatchNorm2d(planes)
        self.relu=ReLU(inplace=True)
        self.conv2=conv3x3(planes,planes)
        self.bn2=BatchNorm2d(planes)
        self.downsample=downsample
        self.stride=stride
    def forward(self,x):
        residual=x

        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)

        if self.downsample is not None:
            residual=self.downsample(x)
        out +=residual
        out = self.relu(out)
        return out
#深层网络（50、101、152）基础块
class Bottleneck(Module):
    expansion =4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1=Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1=BatchNorm2d(planes)
        self.conv2=Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2=BatchNorm2d(planes)
        self.conv3=Conv2d(planes,planes*self.expansion, kernel_size=1,bias=False)
        self.bn3=BatchNorm2d(planes*self.expansion)
        self.relu=ReLU(inplace=True)
        self.downsample=downsample
        self.stride=stride
    def forward(self,x):
        residual=x
        
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu(out)

        out=self.conv3(out)
        out=self.bn3(out)

        if self.downsample is not None:
            residual=self.downsample(x)
        
        out+=residual
        out=self.relu(out)

        return out
    
#模型层级结构
class ResNet(Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes=64
        super(ResNet,self).__init__()

        self.conv1=Conv2d(3,64,kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1=BatchNorm2d(64)
        self.relu=ReLU(inplace=True)
        self.maxpool=MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 定义blocks，第一个卷积层，输入输出维度不变，后三层其中第一块将图像减半，第二次卷积图像大小不变，每层输出通道数加倍
        self.layer1=self._make_layer(block, 64, layers[0])
        self.layer2=self._make_layer(block, 128, layers[1],stride=2)
        self.layer3=self._make_layer(block, 256, layers[0],stride=2)
        self.layer4=self._make_layer(block, 512, layers[0],stride=2)
        #head layer
        self.avgpool=AvgPool2d(7,stride=1)
        self.fc=Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

    def _make_layer(self,block,planes, blocks, stride=1):
        downsample=None
        if stride !=1 or self.inplanes !=planes*block.expansion:
            downsample = nn.Sequential( 
                Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes*block.expansion)                
            )
        layers =[]
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes=planes*block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        
        x=self.avgpool(x)
        x=x.view(x.size(0), -1) # or x=flatten(x,1)  #拉平成1维
        x=self.fc(x)

        return x
#ResNet18
def resnet18(pretrained=False, **kwargs):
    """constructs a ResNet-18 model."""
    model = ResNet(BasicBlock, [2,2,2,2],**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]))
    return model
#ResNet-34
def resnet34(pretrained=False, **kwargs):
    """constructs a ResNet-34 model"""
    model=ResNet(BasicBlock, [3,4,6,3],**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.__loader_url(model_urls["resnet34"]))
    return model
#ResNet-50
def resnet50(pretrained=False, **kwargs):
    """constructs a ResNet-50 model"""
    model=ResNet(Bottleneck, [3,4,6,3],**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.__loader_url(model_urls["resnet50"]))
    return model
#ResNet-101
def resnet101(pretrained=False, **kwargs):
    """constructs a ResNet-101 model"""
    model=ResNet(Bottleneck, [3,4,23,3],**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.__loader_url(model_urls["resnet101"]))
    return model
#ResNet-152
def resnet152(pretrained=False, **kwargs):
    """constructs a ResNet-152 model"""
    model=ResNet(Bottleneck, [3,8,36,3],**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.__loader_url(model_urls["resnet152"]))
    return model


def main():
    #实例化模型
    model = resnet18(pretrained=True)
    new_net_path="./resnet18.onnx"
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model.to(device)
    net.eval()

    #测试数据 batch_size=1, channel=3, image_h=244,image_w=244
    input=torch.randn(1,3, 244,244).to(device)  #BCHW 其中Batch必须为1，因为测试时一般为1，尺寸HW必须和训练时的尺寸一致
    torch.onnx.export(net,input,new_net_path, verbose=False)
    print(model)
    #测试
    with torch.no_grad():
        out=model(input)
        print(out.shape)
if__name__=="__main__":
    main()
