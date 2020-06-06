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
                state['curr_grad'] = None
                    
    def step(self):
        for group in self.param_groups:
            curr_grad = OrderedDict()
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
                curr_grad[k] = g
            return curr_grad

class Node():
    """Node(Choco_Gossip): x_i(t+1) = x_i(t) + gamma*Sum(w_ij*[xhat_j(t+1) - xhat_i(t+1)])"""
    
    def __init__(self, gamma, loader, model, criterion ):
        
        self.neighbors = []

        self.neighbor_wts = {}
        
        self.step_size = gamma
                
        self.dataloader = loader
        
        self.model = model
        
        self.x_i = OrderedDict()
        
        self.model_params = []
        for (k,v) in self.model.state_dict().items():
            
            self.model_params.append(k)
            self.x_i[k] = v.clone().detach()
            
        #for a in self.model.parameters():
        #    self.x_i.append(a)
        
        self.criterion = criterion
        
        self.dataiter = iter(self.dataloader)
        
        self.optimizer = EFSGD(self.model.parameters() , lr = 1e-3 )
                
    
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
        self.curr_gt = self.optimizer.step()
        return
    
    def assign_params(self, W):
        """Assign dict W to model"""
        
        with torch.no_grad():
            self.model.load_state_dict(W, strict=False)
        
        return
    
    def update_model(self):
        
        ### Implement Algorithm ###
        
        ## Assign Parameters after obtaining Consensus##
        
        
        self.assign_params(self.x_i)
        
        return