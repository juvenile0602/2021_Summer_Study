# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms


transform = transforms.Compose(
    [transforms.ToTensor(),#转为Tensor
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])#数据归一化

trainset = torchvision.datasets.CIFAR10(root='Data', train=True,
                                        download=False, transform=transform)
testset = torchvision.datasets.CIFAR10(root='Data', train=False,
                                       download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


(data,label)=trainset[100]


# for i in range(0,len((data,label))):
#     print list1[i];


print(classes[label])