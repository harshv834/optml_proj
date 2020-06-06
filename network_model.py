import autoreload


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



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False, num_workers=2)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')




def trainset_node_split(dataset, N, seed = 0):
    np.random.seed(seed)
    a = np.arange(len(dataset))
    np.random.shuffle(a)
    datasets = {}
    size = int(len(dataset)/N)
    for i in range(N):
        datasets[i] = Subset(dataset, a[i*size:(i+1)*size].tolist())
    return datasets




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



class Node():
    """Node(Choco_Gossip): x_i(t+1) = x_i(t) + gamma*Sum(w_ij*[xhat_j(t+1) - xhat_i(t+1)])"""
    
    def __init__(self, gamma, loader, model, criterion):
        
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
        
    
    def compute_gradient(self, quantizer=None, ):
        """Computes nabla(x_i, samples) and returns estimate after quantization"""
        
        # Sample batch from loader #
        optimizer  = optim.SGD(self.model.parameters(), lr=1e-3)
        #for v in self.model.parameters():
        #  if v.grad is not None:
        #    v.detach_()
        #    v.zero_()

        optimizer.zero_grad()    

        try:
            inputs, targets = self.dataiter.next()
        except StopIteration:
            self.dataiter = iter(self.dataloader)
            inputs, targets = self.dataiter.next()

        
        outputs = self.model(inputs)


        loss = self.criterion(outputs, targets)
        
        #Equivalent to optimizer.zero_grad()
        
        
        loss.backward()
        
        gt = OrderedDict()
        
        
        for k,v in enumerate(self.model.parameters()):
            if v.grad is not None:
                if quantizer is not None:
                    gt[k] = quantizer(v.grad.clone().detach_())
                else:
                    gt[k] = v.grad.clone().detach()
        #optimizer.step()
    
        self.curr_gt = gt
        
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




class Network():
    """Define graph"""
    
    def __init__(self, W, models, learning_rates, loaders, criterion):
        
        self.adjacency = W
        self.num_nodes = W.shape[0]
        
        self.nodes = OrderedDict()
        
        for i in range(self.num_nodes):
            self.nodes[i] = Node(learning_rates[i], loaders[i],models[i], criterion)
            for j in range(self.num_nodes):
                if(j != i and W[i, j] > 0):
                    self.nodes[i].neighbors.append(j)
                    self.nodes[i].neighbor_wts[j] = W[i, j]
                    
            
    def simulate(self, iterations, epochs):
        
        for i in range(epochs):
            for j in range(iterations):
                lr = 1e-3
                if(j % 500 == 0):
                    print(j)
                    
                for k in range(self.num_nodes):
                    self.nodes[k].compute_gradient()
                
                
                for l in range(self.num_nodes):
                    for m,param in enumerate(self.nodes[l].model.parameters()):
                        if param.grad is None:
                            continue
                      
                        gt_update = self.nodes[l].curr_gt[m]
                        wt_sum = 1
                        for n in self.nodes[l].neighbors:
                            gt_update= gt_update + self.nodes[l].neighbor_wts[n] *self.nodes[n].curr_gt[m]
                            wt_sum = wt_sum + self.nodes[l].neighbor_wts[n]
                        gt_update = gt_update/wt_sum
                        param.data = param.data - lr*gt_update
                    #self.nodes[str(l)].update_model()





def count_correct(outputs, labels):
    """ count correct predictions """

    if isinstance(criterion, nn.BCELoss):
        predicted = (outputs > 0.5).to(dtype=torch.int64)
        labels = (labels  > 0.5).to(dtype=torch.int64)
    elif isinstance(criterion, nn.CrossEntropyLoss):
        _, predicted = outputs.max(1)
    else:
        print('Error in criterion')
        raise ValueError

    correct = (predicted == labels).sum().item()

    return correct



def forward_test(model, loader):
    """ forwards test samples and calculates the test accuracy """

    model.to(device)
    running_losses = 0.0
    correct = 0
    total = 0
    count_batches = 0

    with torch.no_grad():

        for batch_idx, sample in enumerate(loader):

            inputs, labels = sample[0].to(device), sample[1].to(device)
            #inputs, labels = sample[0], sample[1]
            outputs = model(inputs)

            loss = criterion(outputs,labels)

            count_batches += 1

            running_losses += loss.item()

            correct += count_correct(outputs, labels)
            total += labels.size(0)


    test_acc  = 100.0 * correct / total
    test_loss = running_losses / total

    print('\n=> Test acc  : {:7.3f}%'.format(test_acc))

    return test_acc, test_loss

def quantizer_topk(gradient, k = 5):
    absoulte = torch.abs( gradient )
    sign  = torch.sign(gradient)
    values,indices = torch.topk( gradient, k , sorted = False )
    gradient = torch.zeros( *gradient.shape )
    gradient[indices] = values
    #transform gradient to torch
    return gradient*sign

def quantizer_lossy( gradient, k = 5 ):
    norm = torch.norm( gradient )
    absoulte = torch.abs( gradient )
    absoulte = ( absoulte/norm )*k
    floor = torch.floor(gradient)
    random_ceil = torch.rand(*gradient.shape) < ( gradient - floor )
    print( random_ceil )
    floor = ( floor + random_ceil.float() ) * (1/k)
    #rescale
    return (norm) * ( torch.sign(gradient) * floor )
    
    
class Agg_Server():
    
    def __init__(self, W, models, clr_model, learning_rates, clr, loaders, criterion):
        
        self.num_nodes = W.shape[0]
        self.server_weights = W
        
        self.central_server = Node(clr, None, clr_model, criterion)
        
        self.nodes = OrderedDict()
        
        for i in range(self.num_nodes):
            self.nodes[i] = Node(learning_rates[i], loaders[i], models[i], criterion)
            
            if(W[i, 0] > 0):
                self.central_server.neighbors.append(i)
                self.central_server.neighbor_wts[i] = W[i, 0]

            
    def simulate(self, iterations, epochs):
        
        for i in range(epochs):
            for j in range(iterations):
                lr = 1e-3
                if(j % 500 == 0):
                    print(j)
                    
                for k in range(self.num_nodes):
                    self.nodes[k].compute_gradient()
    

                for m,param in enumerate(self.central_server.model.parameters()):
                    
                    for a,n in enumerate(self.central_server.neighbors):
                        
                        if self.nodes[n].curr_gt[m] is None:
                            continue
                        
                        if(a == 0):
                            gt = torch.sign(self.nodes[n].curr_gt[m])
                            gt_update = self.central_server.neighbor_wts[n]*gt
                            wt_sum = self.central_server.neighbor_wts[n]
                        else:
                            gt = torch.sign(self.nodes[n].curr_gt[m])
                            gt_update= gt_update + self.central_server.neighbor_wts[n]*gt
                            wt_sum = wt_sum + self.central_server.neighbor_wts[n]
                        
                    gt_update = gt_update/wt_sum
                    param.data = param.data - lr*gt_update
                    
                
                for a,n in enumerate(self.central_server.neighbors):
                    
                    self.nodes[n].model.load_state_dict(self.central_server.model.state_dict())
                
                    #self.nodes[str(l)].update_model()



