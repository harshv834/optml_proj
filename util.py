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

def quantizer_topk(gradient, k = 5):
    absoulte = torch.abs( gradient )
    sign  = torch.sign(gradient)
    values,indices = torch.topk( gradient, k , sorted = False )
    gradient = torch.zeros( *gradient.shape )
    gradient[indices] = values
    #transform gradient to torch
    return gradient*sign

def quantizer_lossy( gradient, k = 64 ):
    norm = torch.norm( gradient )
    absoulte = torch.abs( gradient )
    absoulte = ( absoulte/norm )*k
    floor = torch.floor(absoulte)
    random_ceil = torch.rand(*gradient.shape) < ( gradient - floor )
    print( random_ceil )
    floor = ( floor + random_ceil.float() ) * (1/k)
    #rescale
    return (norm) * ( torch.sign(gradient) * floor )

def ring( num_workers ):
	ring = torch.zeros([num_workers, num_workers])
	for i in range(num_workers-1):
	    ring[i,i+1] = 1.0
	    ring[i,i-1] = 1.0
	#close
	ring[num_workers - 1, 0 ] = 1.0
	ring[num_workers - 1, num_workers-2 ] = 1.0
	return ring

def torus( sqrt_num_workers):
	num_workers = sqrt_num_workers*sqrt_num_workers
	torus = torch.zeros([num_workers, num_workers])
	for i in range(num_workers):
		row = i / sqrt_num_workers
		column  = i % sqrt_num_workers
		neighbour = [-1,0,1]
		for a in neighbour:
			if( row + a < 0 or row + a == sqrt_num_workers ):
				continue
			if( r+1 > )



def degree_k( num_worker , k ):


