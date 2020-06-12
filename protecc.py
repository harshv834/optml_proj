
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
from torch.utils.data import Subset


#Majority Vote
#https://arxiv.org/pdf/1810.05291.pdf

def get_vote(grads):
    """Grads is a list of vectors coming from the node and its neighbors only, grads[0] = node.grad"""
    V = torch.zeros_like(grads[0])
    
    for i in grads:
        V+=torch.sign(i.clone().detach())
    
    return V

## Final update on every worker as w = w - neta*(V + lambda*w)(lambda = regularization parameter)




#Robust SGD, Robust One Round
#https://arxiv.org/pdf/1803.01498.pdf


def get_statistic(grads, option = 1, beta = 1/3):
    """option=1 == median, option = 2 == mean"""
    
    
    V = torch.stack(grads, dim=0)
    
    if(option ==1):
        values, indices = torch.median(V, dim=0)
        temp = values.clone().detach()
    else:
        m = torch.sort(V, dim=0)[0].clone().detach()
        first_index = int(beta*m.size()[0])
        last_index = int((1-beta)*m.size()[0])
        
        total = last_index - first_index
        
        temp = torch.zeros_like(grads[0])
        
        if(total > 0):
            for i in range(total):
                temp+=m[i+first_index]

            temp = temp/total
    
    return temp.clone().detach()



def get_frac(grads, beta = 1/3):
    V = torch.stack(grads, dim=0)
    gradnorm,_ = torch.sort(torch.norm(V,dim=0).clone().detach())
    print(gradnorm.shape)
    temp = gradnorm[:int((1-beta)*len(grads))].mean()
    return temp
