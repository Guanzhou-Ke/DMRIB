import torch
from torch import nn
from torchvision.models.resnet import (resnet18, 
                                       resnet34, 
                                       resnet50,
                                       resnet101,
                                       resnet152)



class SmallResNet(nn.Module):
    
    def __init__(self, model, channels=3):
        super(SmallResNet, self).__init__()

        self.f = []
        for name, module in model.named_children():
            if name == 'conv1':
                module = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # # projection head
        # self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
        #                        nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        # out = self.g(feature)
        return feature


class RegualrResNet(nn.Module):
    
    def __init__(self, model, channels=3):
        super(RegualrResNet, self).__init__()

        self.f = []
        for name, module in model.named_children():
            if name == 'conv1':
                module = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if not isinstance(module, nn.Linear):
                self.f.append(module)
        self.f = nn.Sequential(*self.f)
        
    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        # out = self.g(feature)
        return feature


def _resnet(model_func, channels, maxpooling):
    model = model_func()
    if maxpooling:
        model = RegualrResNet(model, channels=channels)
    else:
        model = SmallResNet(model, channels=channels)
    return model


def ResNet18(channels=3, maxpooling=True, **kwargs):
    return _resnet(resnet18, channels=channels, maxpooling=maxpooling)        

def ResNet34(channels=3, maxpooling=True, **kwargs):
    return _resnet(resnet34, channels=channels, maxpooling=maxpooling)        

def ResNet50(channels=3, maxpooling=True, **kwargs):
    return _resnet(resnet50, channels=channels, maxpooling=maxpooling)        

def ResNet101(channels=3, maxpooling=True, **kwargs):
    return _resnet(resnet101, channels=channels, maxpooling=maxpooling)        

def ResNet152(channels=3, maxpooling=True, **kwargs):
    return _resnet(resnet152, channels=channels, maxpooling=maxpooling)        


    
if __name__ == '__main__':
    pass
    