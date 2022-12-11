import numpy as np
import time
from function import *
from l2_weights import *
from majorityVoting import *
from model import model as mod
from scipy.special import xlogy
from sklearn.metrics import log_loss,mean_squared_error
import tensorflow as tf
import math

NUM=1      # possible target number
N = 16  # array number
phi = 32
f = 500
c = 1500
lmbda = c / f  # wave length
k=2*np.pi*f/c   # wavenumber
Lr=np.linspace(0,1,N)  # array length


def cusLoss(origin,r):
    r=np.reshape(r,(1,np.size(r)))
    xr=np.zeros([N,])
    xr_i=np.zeros([N,])
    for n in range(NUM):
        xr=xr+r[0,n]*tf.cos(k*tf.cos(180*r[0,n+NUM]*math.pi/180)*Lr+2*math.pi*r[0,n+2*NUM])
        xr_i=xr_i+r[0,n]*tf.sin(k*tf.cos(180*r[0,n+NUM]*math.pi/180)*Lr+2*math.pi*r[0,n+2*NUM])
    loss=(tf.norm(origin[0][0:N]-xr,ord=2)+tf.norm(origin[0][N:2*N]-xr_i,ord=2))/10+0.0*tf.norm(r[0,0:(NUM-1)],ord=1)
    return loss

def RVFLvtrain(trainX,trainY,option):
   
    
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
            w = s * 2 * rand_seed.rand(n_dims,N) - 1

        else:
            #w = s*2*rand_seed.rand(n_dims+N,N)-1
            w = s * 2 * rand_seed.rand(N, N) - 1

        b = s*rand_seed.rand(1, N)
        weights.append(w)
        biases.append(b)
        A_ = np.matmul(A_input, w)
       
        # layer normalization
        A1_mean = np.mean(A_)
        A1_std = np.std(A_)
        A_ = (A_-A1_mean)/A1_std

        mu.append(A1_mean)
        sigma.append(A1_std)

        A_ = A_ + np.repeat(b, n_sample, 0)
        A_ = relu(A_)
        #A_ = selu(A_)
        # trainX *= ada_weight * n_sample
        #A_tmp = np.concatenate([trainX,A_,np.ones((n_sample,1))],axis=1)
        A_tmp = np.concatenate([A_, np.ones((n_sample, 1))],axis=1)
        
        n_classes = 3*NUM # np.unique 去除其中重复的元素 ，并按元素 由小到大
        outY = np.hstack((Am,theta,phi))
        outY = np.reshape(outY,(1,np.size(outY)))
        beta_=l2_weights(A_tmp,outY,C,n_sample)

        A.append(A_tmp)
        beta.append(beta_)

        #clear A_ A_tmp A1_temp2 beta_

        trainY_temp=np.matmul(A_tmp,beta_)
        
        prob = np.expand_dims(softmax(trainY_temp), axis=1)
        h = (n_classes - 1) * (np.log(prob) - (1. / n_classes) * np.log(prob).sum(axis=1)[:, np.newaxis])
        #indx=np.argmax(trainY_temp,axis=1)
        #indx=indx.reshape(n_sample,1)
        y_codes = np.array([-1. / (n_classes - 1), 1.])
        y_coding = y_codes.take(np.unique(outY) == outY[:, np.newaxis])
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
    pred = pred.ravel()
    
    
    time_end = time.time()
    Training_time = time_end-time_start
    


    # Training_loss = mean_squared_error(trainY, reCons)
    Training_loss = cusLoss(trainY,pred)
    print('loss:',Training_loss)
    model = mod(L,weights,biases,beta,mu,sigma,ada_weights,n_types)
        
    return model,Training_time,Training_loss

