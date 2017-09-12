#! /usr/bin/env python
#################################################################################
#     File Name           :     bp.py
#     Created By          :     hzr
#     Creation Date       :     [2017-04-20 18:10]
#     Last Modified       :     [2017-04-25 16:32]
#     Description         :
#################################################################################

import numpy as np
from numpy import linalg as LA
from numpy import random as rdm
import os

import time
import pdb

def getvaluespart(featurefile,batch_size):
        s = []
        no_of_features = 516
        fh = featurefile
        while batch_size:
                line = fh.readline().strip()
                if len(line) == 0:
                    break
                parts = line.split()
                assert (len(parts) == no_of_features)
                buff = [float(v) for v in parts[4:]]
                s.append(buff)
                batch_size-=1

        return np.array(s)

def forward(batchsize):
	s = getvaluespart(fh,batchsize)
 #       s = s / np.sqrt(np.sum(s * s,axis=1))[:,None]

	h, w = s.shape
	#covariance, mean = getCovariance(s, h, w)
	mean = np.mean(s,axis=0)
	#covariance = np.cov(s,rowvar=0)
	return mean,s

def backward(W_,mean_,batchsize,cnt):
	mean,features = forward(batchsize)

	lmda= 0.001
	lr=0.001
	#global mean_of_batches
	#pdb.set_trace()
	inner_product = np.dot((features - mean_).transpose(),(features - mean))
	tmp = np.dot(inner_product/batchsize,W_)
	constrained_gradient=tmp + lmda*np.dot((np.dot(W_,W_.transpose())-np.identity(512)),W_)
	#print np.dot((np.dot(W_,W_.transpose())-np.identity(512)),W_)[0][0]

	# if(np.log10(constrained_gradient)[0][0]>100):
	# 	pdb.set_trace()
	print "------------------"

	W_ = W_ + lr*constrained_gradient
	#print mean_
	#************************************
	mean_ = (mean + (cnt-1)*mean_)/cnt
	#print cov_
	#covMatInv = np.linalg.inv(cov_)
	#************************************
	return W_,mean_

start = time.clock()
cov = np.identity(512)
W = rdm.random(size=(512,256))
mean = np.zeros((512,))
#global mean_of_batches
#mean_of_batches = np.empty((0,512,))
filePath = "/home/hzr/work/UCSDfused/VGGplaces/conv4_1_train_feature_list.txt"
with open(filePath, "r") as fh:
	iter_time = 10
	a = iter_time
	while a:
	    W ,mean = backward(W,mean,1024,iter_time-a+1)#last W and mean, batch size, iteration times
	    print W[0][0]#print the first element in the decomposed covarience matrix to see whether it becomes too large
	    a-=1
            if a % 100 == 0:
                print a, ' iterations remaining'
	#mean = np.mean(mean_of_batches,axis=0)
end = time.clock()
print 'time:'+str(end - start)+' s'
print mean.shape
print W.shape
print "saving..."
filename_cov = "/home/hzr/work/UCSDfused/VGGplaces/bp_learn/learned_mean_cov/lowrank_W_.txt"
filename_mean = "/home/hzr/work/UCSDfused/VGGplaces/bp_learn/learned_mean_cov/lowrank_mean_.txt"

np.savetxt(filename_cov , W, fmt = "%6f")
np.savetxt(filename_mean , mean, fmt = "%6f")
