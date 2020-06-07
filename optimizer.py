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
from torch.optim.optimizer import Optimizer
from model_util import *

class EFSGD(Optimizer):
    def __init__(self, params, lr ):
        super(EFSGD,self).__init__( params , dict( lr = lr ) )
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['error_correction'] = torch.zeros_like( param.data )
                state['lr'] = lr
                
                    
    def step(self):
        for group in self.param_groups:
            for k,param in enumerate(group['params']):
                if param.grad is None:
                    continue
                state = self.state[param] 
                error_corr = state['error_correction']
                lr = state['lr']
                p = param.grad.data
                p = lr*p + error_corr 
                
                #EFSGD
                g = ( torch.sum( torch.abs(p) )/p.nelement() ) * torch.sign(p)
                state['error_correction'] = p - g
                state['update'] = g


class signSGD(Optimizer):
    def __init__(self, params, lr ):
        super(EFSGD,self).__init__( params , dict( lr = lr ) )
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['lr'] = lr
                
                    
    def step(self):
        for group in self.param_groups:
            for k,param in enumerate(group['params']):
                if param.grad is None:
                    continue
                state = self.state[param] 
                lr = state['lr']
                state['update'] = lr*param.grad.data.sign()

class QSGD_lossy(Optimizer):
    def __init__(self, params, lr ):
        super(EFSGD,self).__init__( params , dict( lr = lr ) )
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['lr'] = lr
                
                    
    def step(self):
        for group in self.param_groups:
            for k,param in enumerate(group['params']):
                if param.grad is None:
                    continue
                state = self.state[param] 
                lr = state['lr']
                state['update'] = lr*quantizer_lossy(param.grad.data)


class QSGD_topk(Optimizer):
    def __init__(self, params, lr ):
        super(EFSGD,self).__init__( params , dict( lr = lr ) )
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['lr'] = lr
                
                    
    def step(self):
        for group in self.param_groups:
            for k,param in enumerate(group['params']):
                if param.grad is None:
                    continue
                state = self.state[param] 
                lr = state['lr']
                state['update'] = lr*quantizer_topk(param.grad.data)

                
class QEFSGD_lossy(Optimizer):
    def __init__(self, params, lr,beta,alpha):
        super(EFSGD,self).__init__( params , dict( lr = lr, beta = beta,alpha=alpha) )
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['error_correction'] = torch.zeros_like( param.data )
                state['lr'] = lr
                state['beta'] =beta
                state['alpha'] =alpha
                    
    def step(self):
        for group in self.param_groups:
            for k,param in enumerate(group['params']):
                if param.grad is None:
                    continue
                state = self.state[param] 
                lr = state['lr']
                state['update'] = lr*quantizer_lossy(state['error_correction']*state['alpha'] + param.grad.data)
                state['error_correction'] = beta*state['error_correction'] - state['update'] +param.grad.data 
                

class QEFSGD_topk(Optimizer):
    def __init__(self, params, lr,beta,alpha):
        super(EFSGD,self).__init__( params , dict( lr = lr, beta = beta,alpha=alpha) )
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['error_correction'] = torch.zeros_like( param.data )
                state['lr'] = lr
                state['beta'] =beta
                state['alpha'] =alpha
                    
    def step(self):
        for group in self.param_groups:
            for k,param in enumerate(group['params']):
                if param.grad is None:
                    continue
                state = self.state[param] 
                lr = state['lr']
                state['update'] = lr*quantizer_topk(state['error_correction']*state['alpha'] + param.grad.data)
                state['error_correction'] = beta*state['error_correction'] - state['update'] +param.grad.data 
                
