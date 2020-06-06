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
	
	def __init__(self, W, models, learning_rates, loaders, criterion, byz ):
		
		self.adjacency = W
		self.num_nodes = W.shape[0]
		
		self.nodes = OrderedDict()
		
		for i in range(self.num_nodes):
			isbyz = i in byz
			self.nodes[i] = Node(learning_rates[i], loaders[i],models[i], criterion, isbyz )
			for j in range(self.num_nodes):
				if(j != i and W[i, j] > 0):
					self.nodes[i].neighbors.append(j)
					self.nodes[i].neighbor_wts[j] = W[i, j]
					
			
	def simulate(self, iterations, epochs):
		
		for i in range(epochs):
			for j in range(iterations):
				if(j % 500 == 0):
					print(j)
					
				for k in range(self.num_nodes):
					self.nodes[k].compute_gradient()
				
				
				for l in range(self.num_nodes):
					if self.node[l].isbyz :
						self.update_byz( l )
						continue
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

	def update_byz( self, index ):
		for m,param in enumerate(self.nodes[index].model.parameters()):
			if param.grad is None:
				continue
		  
			gt_update = self.nodes[index].curr_gt[m].copy()
			wt_sum = 1
			for n in self.nodes[n].neighbors:
				gt_update= gt_update + self.nodes[index].neighbor_wts[n] *self.nodes[n].curr_gt[m]
				wt_sum = wt_sum + abs( self.nodes[index].neighbor_wts[n] )
			gt_update = gt_update/wt_sum
			param.grad.data = gt_update
		self.node[index].update_model()
		return



