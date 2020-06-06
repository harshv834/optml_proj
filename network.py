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

class Network():
    """Define graph"""
    
    def __init__(self, W, models, datasets, learning_rates, loaders, batch_size, criterion, chosen_device, testloader):
        self.adjacency = W
        self.num_nodes = W.shape[0]
        self.chosen_device = chosen_device
        self.batch_size = batch_size
        self.testloader = testloader
        
        self.nodes = OrderedDict()
        
        for i in range(self.num_nodes):
            self.nodes[i] = Node(learning_rates[i], loaders[i], self.batch_size, datasets[i], models[i], criterion, self.chosen_device)
            for j in range(self.num_nodes):
                if(j != i and W[i, j] > 0):
                    self.nodes[i].neighbors.append(j)
                    self.nodes[i].neighbor_wts[j] = W[i, j]
                    
            
    def consensus_test(self, loader, batch_size):
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
                    
                    for i in range(batch_size):
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

        
        for i in range(epochs):
            for j in range(iterations):
                if((j+1) % 500 == 0 and j != 0):
   
                  test_acc = self.consensus_test(self.testloader, self.batch_size)
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

    def update_network(self):            
        for l in range(self.num_nodes):
            for m,param in enumerate(self.nodes[l].model.parameters()):
                if param.grad is None:
                    continue
                      
            gt_update = self.nodes[l].curr_gt[m].copy()
            wt_sum = 1
            for n in self.nodes[l].neighbors:
                gt_update= gt_update + self.nodes[l].neighbor_wts[n] *self.nodes[n].curr_gt[m]
                wt_sum = wt_sum + abs( self.nodes[l].neighbor_wts[n] )
            gt_update = gt_update/wt_sum
            param.grad.data = gt_update
        self.node[l].update_model()


class Node():
    """Node(Choco_Gossip): x_i(t+1) = x_i(t) + gamma*Sum(w_ij*[xhat_j(t+1) - xhat_i(t+1)])"""
    
    def __init__(self, gamma, loader, batch_size, dataset, model, criterion, chosen_device):
        
        self.neighbors = []

        self.neighbor_wts = {}
        self.step_size = gamma

        self.dataset = dataset

        self.dataloader = loader
        
        self.model = model
        self.chosen_device = chosen_device
        
        
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
        for k,v in enumerate(self.model.parameters()):
            if v.grad is not None:
                if quantizer is not None:
                    gt[k] = quantizer(v.grad)
                else:
                    gt[k] = v.grad
        self.curr_gt =  gt
        return
    
    def assign_params(self, W):
        """Assign dict W to model"""
        
        with torch.no_grad():
            self.model.load_state_dict(W, strict=False)
        
        return
    
    def update_model(self):
        ## Assign Parameters after obtaining Consensus##
        self.optimizer.step()        
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
                
                loss = criterion(outputs,labels)
                
                count_batches += 1

                running_losses += loss.item()

                correct += count_correct(outputs, labels)
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
                
                loss = criterion(outputs,labels)
                
                count_batches += 1

                running_losses += loss.item()

                correct += count_correct(outputs, labels)
                total += labels.size(0)

                running_losses = running_losses/self.batch_size


        loss_dict["test_acc"]  = 100.0 * correct / total
        loss_dict["test_loss"] = running_losses / total

        #print('\n=> Test acc  : {:7.3f}%'.format(test_acc))

        return loss_dict