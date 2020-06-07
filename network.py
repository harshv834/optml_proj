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
from optimizer import *
from model_util import *
from config import *
import tqdm

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        #self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        #self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 4 * 4, 64)
        #self.fc2 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(64, 10)
        self.init_weights()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x

    def init_weights(self):
        
        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out',nonlinearity='relu')

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
                torch.nn.init.xavier_uniform_(m.weight)

class Network():
    """Define graph"""
    
    def __init__(self, W, models, datasets, learning_rates, loaders, batch_size, criterion, chosen_device, testloader, optimizer, byz_nodes = [] , attack_mode = "" ):
        self.adjacency = W
        self.num_nodes = W.shape[0]
        self.chosen_device = chosen_device
        self.batch_size = batch_size
        self.testloader = testloader
        
        self.nodes = OrderedDict()
        
        for i in range(self.num_nodes):
            isbyn = i in byz_nodes
            self.nodes[i] = Node(learning_rates[i], loaders[i], self.batch_size, datasets[i], models[i], criterion, self.chosen_device, optimizer, isbyn , attack_mode )
            for j in range(self.num_nodes):
                if(j != i and W[i, j] > 0):
                    self.nodes[i].neighbors.append(j)
                    self.nodes[i].neighbor_wts[j] = W[i, j]
                    
            
    def consensus_test(self, loader):
        """ forwards test samples and calculates the test accuracy """

        #model.to(chosen_device)
        correct = 0
        total = 0
        count_batches = 0

        with torch.no_grad():

            for batch_idx, sample in enumerate(loader):
                inputs, labels = sample[0].to(self.chosen_device), sample[1].to(self.chosen_device)
                #inputs, labels = sample[0], sample[1]
                for i in range(self.num_nodes):
                    self.nodes[i].model.to(self.chosen_device)
                    outputs = self.nodes[i].model(inputs)
                    if(i == 0):
                        consensus = torch.zeros_like(outputs)

                    logits, predicted = outputs.max(1)
                    #print(logits)
                    
                    for i in range(labels.size(0)):
                        consensus[i] = consensus[i]+torch.where(outputs[i].eq(logits[i]), torch.Tensor([1]).to(self.chosen_device), torch.Tensor([0]).to(self.chosen_device))              
                
                logits, final_pred = consensus.max(1)

                batch_correct = (final_pred == labels).sum().item()
                
                correct += batch_correct
                total += labels.size(0)

        test_acc  = 100.0 * correct / total
        #test_loss = running_losses / total

        #print('\n=> Test acc  : {:7.3f}%'.format(test_acc))

        return test_acc
    
    def simulate(self, iterations, epochs):
        record_sims = OrderedDict()
        for k in range(self.num_nodes):
            record_sims[k] = []

        
        for i in tqdm.tqdm(range(epochs)):
            for j in tqdm.tqdm(range(iterations)):
                if((j+1) % 500 == 0 and j != 0):
   
                  test_acc = self.consensus_test(self.testloader)
                  for k in range(self.num_nodes):
                    loss_dict = self.nodes[k].calc_node_loss(self.testloader, self.chosen_device)
                    loss_dict["consensus_test"] = test_acc
                    loss_dict["iteration"] = j
                    record_sims[k].append(loss_dict)
                if(j % 500 == 0):
                    print(j)

                for k in range(self.num_nodes):
                    self.nodes[k].compute_gradient()
                
                self.attack()
                self.update_network()
        return record_sims

    def update_network(self): 
        for l in range(self.num_nodes):
            for group_id in range(len(self.nodes[l].optimizer.param_groups)):
                for m,param in enumerate(self.nodes[l].optimizer.param_groups[group_id]['params']):
                    if param.grad is None:
                        continue
                    
                    if self.nodes[l].isbyn :
                        # if it is a byzantine load then update as per original.
                        gt_update = self.nodes[l].orig_gt[group_id][m].clone()
                    else: 
                        gt_update = self.nodes[l].curr_gt[group_id][m].clone()
                    wt_sum = 1
                    for n in self.nodes[l].neighbors:
                        gt_update= gt_update + self.nodes[l].neighbor_wts[n] *self.nodes[n].curr_gt[group_id][m]
                        wt_sum = wt_sum + abs( self.nodes[l].neighbor_wts[n] )
                    gt_update = gt_update/wt_sum
                    param.grad.data -= gt_update
        
        
    def attack(self):
        #attack based on gradients of neighbours.
        return

class Node():
    """Node(Choco_Gossip): x_i(t+1) = x_i(t) + gamma*Sum(w_ij*[xhat_j(t+1) - xhat_i(t+1)])"""
    
    def __init__(self, gamma, loader, batch_size, dataset, model, criterion, chosen_device, optimizer , isbyn = False, attack_mode = ""):
        
        self.neighbors = []

        self.neighbor_wts = {}
        self.step_size = gamma

        self.dataset = dataset

        self.dataloader = loader

        self.isbyn = isbyn

        assert  attack_mode in [ "", "full_reversal","random_reversal" ] 

        self.attack_mode = attack_mode

        
        self.model = model
        self.chosen_device = chosen_device
        self.batch_size = batch_size
        
        self.x_i = OrderedDict()
        self.model_params = []
        for (k,v) in self.model.state_dict().items():
            
            self.model_params.append(k)
            self.x_i[k] = v.clone().detach()
            
        #for a in self.model.parameters():
        #    self.x_i.append(a)
        
        self.criterion = criterion
        
        self.dataiter = iter(self.dataloader)
        
        self.optimizer = optimizer(self.model.parameters() , lr = self.step_size )

        #broadcast
        self.curr_gt = None

        #original gradient for self update
        self.orig_gt = None
                
    
    def compute_gradient(self):
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
        
        self.optimizer.step()
        gt = []
        for group in self.optimizer.param_groups:
            param_update = OrderedDict()
            for k,param in enumerate(group['params']):
                if param.grad is None:
                    continue
                state = self.optimizer.state[param]
                param_update[k] = state['update'].clone().detach()
            gt.append(param_update)    
        self.curr_gt =  gt
        if self.isbyn and self.attack_mode != "":
            self.attack()
        return

    def attack(self):
        self.orig_gt = []
        for group_grad in self.curr_gt:
            orig = OrderedDict()
            for key,grad in group_grad.items():
                orig[key] = grad.clone().detach()
                if self.attack_mode == "full_reversal" :
                    grad = grad*-1
                elif self.attack_mode == "random_reversal" :
                    rev = torch.rand(*grad.shape) < RANDOM_REV 
                    sign_rev = torch.sign(grad) * ( 1 + rev.float()*-2 )
                    grad = grad * sign_rev 
                else:
                    continue
            self.orig_gt.append(orig)

    def assign_params(self, W):
        """Assign dict W to model"""
        
        with torch.no_grad():
            self.model.load_state_dict(W, strict=False)
        
        return
    
    
    def calc_node_loss(self, testloader, chosen_device):
        """ loss check """
        
        loss_dict = {}

        #self.model.to(chosen_device)
        trainloader = torch.utils.data.DataLoader(self.dataset, self.batch_size, shuffle=True, num_workers=2)

        running_losses = 0.0
        correct = 0
        total = 0
        count_batches = 0

        with torch.no_grad():

            for batch_idx, sample in enumerate(trainloader):

                inputs, labels = sample[0].to(chosen_device), sample[1].to(chosen_device)
                #inputs, labels = sample[0], sample[1]
                self.model.to(self.chosen_device)
                outputs = self.model(inputs)
                
                loss = self.criterion(outputs,labels)
                
                count_batches += 1

                running_losses += loss.item()

                correct += count_correct(outputs, labels,self.criterion)
                total += labels.size(0)

                running_losses = running_losses/self.batch_size


        loss_dict["train_acc"]  = 100.0 * correct / total
        loss_dict["train_loss"] = running_losses / total

        running_losses = 0.0
        correct = 0
        total = 0
        count_batches = 0

        with torch.no_grad():

            for batch_idx, sample in enumerate(testloader):

                inputs, labels = sample[0].to(chosen_device), sample[1].to(chosen_device)
                #inputs, labels = sample[0], sample[1]
                self.model.to(self.chosen_device)
                outputs = self.model(inputs)
                
                loss = self.criterion(outputs,labels)
                
                count_batches += 1

                running_losses += loss.item()

                correct += count_correct(outputs, labels, self.criterion)
                total += labels.size(0)

                running_losses = running_losses/self.batch_size


        loss_dict["test_acc"]  = 100.0 * correct / total
        loss_dict["test_loss"] = running_losses / total

        #print('\n=> Test acc  : {:7.3f}%'.format(test_acc))

        return loss_dict