import numpy as np
import time
from function import *
from l2_weights import *
from majorityVoting import *
from model import model as mod
from scipy.special import xlogy
from sklearn.metrics import log_loss
import tensorflow as tf
import math

global NUM
NUM= 10      # maximum possible targets
Lr=np.linspace(0,10,21)  # array length
f = 500
c = 1500
lmbda = c / f  # wave length
k=2*np.pi*f/c   # wave number

def customLoss(origin,r):
    xr=tf.zeros([21,])
    xr_i=tf.zeros([21,])
    for n in range(NUM):
        xr=xr+r[0,n]*tf.cos(k*tf.cos(180*r[0,n+NUM]*math.pi/180)*Lr+2*math.pi*r[0,n+2*NUM])
        xr_i=xr_i+r[0,n]*tf.sin(k*tf.cos(180*r[0,n+NUM]*math.pi/180)*Lr+2*math.pi*r[0,n+2*NUM])
    loss=(tf.norm(origin[0][0:21]-xr,ord=2)+tf.norm(origin[0][21:42]-xr_i,ord=2))/10+0.001*tf.norm(r[0,0:(NUM-1)],ord=1)
    return loss

def SSELMLPtrain(trainX,trainY,data,option):
    
    rand_seed = np.random.RandomState(option.seed)

    [n_sample, n_dims] = trainX.shape
    N = option.N
    L = option.L
    C = option.C
    s = option.scale   #scaling factor
    A=[]
    beta=[]
    weights = []
    biases = []
    mu = []
    sigma = []
    samm_prob = []

    A_input= trainX
    n_types = option.n_types
    time_start=time.time()
    ada_weights = np.ones((L, n_sample))
    ada_weight = np.expand_dims(np.ones(len(trainX))/len(trainX), axis=1)

    for i in range(L):

        if i==0:
            w = s*2*rand_seed.rand(n_dims,N)-1

        else:
            #w = s*2*rand_seed.rand(n_dims+N,N)-1
            w = s * 2 * rand_seed.rand(N, N) - 1

        b = s*rand_seed.rand(1, N)
        weights.append(w)
        biases.append(b)
        # print(A_input.shape)
        # print(w.shape)

        A_ = np.matmul(A_input, w)

        # layer normalization
        A1_mean = np.mean(A_, axis=0)
        A1_std = np.std(A_, axis=0)
        A_ = (A_-A1_mean)/A1_std
        mu.append(A1_mean)
        sigma.append(A1_std)

        A_ = A_ + np.repeat(b, n_sample, 0)
        A_ = relu(A_)
        #A_ = selu(A_)
        # trainX *= ada_weight * n_sample
        #A_tmp = np.concatenate([trainX,A_,np.ones((n_sample,1))],axis=1)
        A_tmp = np.concatenate([A_, np.ones((n_sample, 1))],axis=1)
        beta_=l2_weights(A_tmp,trainY,C,n_sample)

        A.append(A_tmp)
        beta.append(beta_)

        #clear A_ A_tmp A1_temp2 beta_

        trainY_temp=np.matmul(A_tmp,beta_)
        n_classes = np.unique(trainY).size
        prob = np.expand_dims(softmax(trainY_temp), axis=1)
        h = (n_classes - 1) * (np.log(prob) - (1. / n_classes) * np.log(prob).sum(axis=1)[:, np.newaxis])
        #indx=np.argmax(trainY_temp,axis=1)
        #indx=indx.reshape(n_sample,1)
        y_codes = np.array([-1. / (n_classes - 1), 1.])
        y_coding = y_codes.take(np.unique(trainY) == trainY[:, np.newaxis])
        estimator_weight = (-1.
                            * ((n_classes - 1.) / n_classes)
                            * xlogy(y_coding, prob).sum(axis=2))
        ada_weight *= np.exp(estimator_weight *((ada_weight > 0) |(estimator_weight < 0)))
        ada_weight /= ada_weight.sum()
        samm_prob.append(h)
        ada_weights[i] = ada_weight.ravel()

        # trainX *= ada_weight*n_sample
        # A_input = np.concatenate([trainX, A_], axis=1)
        A_input = A_


    pred = sum(samm_prob) / L

    time_end = time.time()
    Training_time = time_end-time_start


    ## Calculate the training accuracy

    #TrainingAccuracy = np.sum(np.argmax(pred, axis=2).ravel() == np.argmax(trainY,axis=1).ravel())/n_sample

    pred = np.argmax(pred, axis=2).ravel()
    TrainingAccuracy = np.sum(pred == np.argmax(trainY, axis=1).ravel()) / n_sample
    # print(np.argmax(pred, axis=2).ravel())
    # print(len(trainY))
   
    Training_loss = customLoss(data,data)

    model = mod(L,weights,biases,beta,mu,sigma,ada_weights,n_types)
        
    return model,TrainingAccuracy,Training_time,Training_loss

