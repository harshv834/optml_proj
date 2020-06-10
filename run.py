import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import time
import os
from collections import OrderedDict
from torch.utils.data import Subset, Dataset
from sklearn.datasets import fetch_rcv1
import tqdm
from network import *
from optimizer import *
from model_util import *
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=8,shuffle=False, num_workers=2)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')


criterion = nn.CrossEntropyLoss()
sqrt_num_workers = 3
num_workers = 9
m = trainset_node_split(trainset, num_workers)
trainloaders = [torch.utils.data.DataLoader(m[i], batch_size=32, shuffle=True, num_workers=2) for i in range(num_workers)]
W = ring(num_workers)
lrs = []


for i in range(num_workers):
    # lrs.append({'lr':1e-3,'beta':0.9,'alpha':0.1})
    lrs.append({'lr':1e-3})

models = [Net() for i in range(num_workers)]

attacks = ['full_reversal','random_reversal']
protecs = ['median','trmean','majority',None]
net = Network(W, models, m, lrs, trainloaders, 32, nn.CrossEntropyLoss(), device, testloader, signSGD, [3,5], "full_reversal", None)

a = net.simulate(20000, 1)

pickle.dump(a,open("ring_signsgd_fullr_2node.pickle","wb"))