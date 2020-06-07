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

class EFSGD(Optimizer):
    def __init__(self, params, lr ):
        super(EFSGD,self).__init__( params , dict( lr = lr ) )
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['error_correction'] = torch.zeros_like( param.data )
                state['lr'] = lr
                state['update'] = OrderedDict()
                
                    
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
                state['update'][k] = g

<<<<<<< HEAD
class Node():
    """Node(Choco_Gossip): x_i(t+1) = x_i(t) + gamma*Sum(w_ij*[xhat_j(t+1) - xhat_i(t+1)])"""
    
    def __init__(self, gamma, loader, model, criterion, isbyz ):
        
        self.neighbors = []

        self.isbyz = isbyz

        self.neighbor_wts = {}
        
        self.step_size = gamma
=======
class QEFSGD(Optimizer):
    def __init__(self, params, lr ):
        super(EFSGD,self).__init__( params , dict( lr = lr ) )
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['error_correction'] = torch.zeros_like( param.data )
                state['lr'] = lr
>>>>>>> 3b8b959da766bd496c24acdeef0cb208945617d5
                
                    
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
                
<<<<<<< HEAD
    
    def compute_gradient(self, quantizer=None, ):
        """Computes nabla(x_i, samples) and returns estimate after quantization"""        
        self.optimizer.zero_grad() 
        try:
            inputs, targets = self.dataiter.next()
        except StopIteration:
            self.dataiter = iter(self.dataloader)
            inputs, targets = self.dataiter.next()
        output = self.model(inputs)
        loss = self.criterion(output, targets)
        loss.backward()
        for k,v in enumerate(self.model.parameters()):
            if v.grad is not None:
                if quantizer is not None:
                    gt[k] = quantizer(v.grad)
                else:
                    gt[k] = v.grad
            if( self.isbyz ):
                gt[k] = self.attack(gt)
        self.curr_gt =  gt
        return

    def attack(grad):
        #TODO
        return grad
    
    def assign_params(self, W):
        """Assign dict W to model"""
        
        with torch.no_grad():
            self.model.load_state_dict(W, strict=False)
        
        return
    
    def update_model(self):
        ## Assign Parameters after obtaining Consensus##
        self.optimizer.step()        
        return
=======
                #EFSGD
                g = ( torch.sum( torch.abs(p) )/p.nelement() ) * torch.sign(p)
                state['error_correction'] = p - g
                param.data = param.data - g
>>>>>>> 3b8b959da766bd496c24acdeef0cb208945617d5
