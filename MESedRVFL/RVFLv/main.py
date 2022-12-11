import os
import numpy as np
from option import option as op
from RVFLv import *
import time
import scipy.io as sio
from Logger import Logger
import sys
global NUM
np.seterr(divide='ignore', invalid='ignore')



NUM=1      # possible target number
N = 16  # array number
phi = 32
f = 500
c = 1500
lmbda = c / f  # wave length
k=2*np.pi*f/c   # wavenumber
Lr=np.linspace(0,1,N)  # array length
Am = 1
theta = 20
phi = 0

Pr=Am*np.cos(Lr*k*np.cos(theta*np.pi/180+phi))  # signal real
Pr_i=Am*np.sin(Lr*k*np.cos(theta*np.pi/180+phi))  # signal imaginary
signal = np.hstack((Pr,Pr_i))
x_dnn=np.reshape(signal,(1,2*N))

def get_avg(list):
    sum = 0
    for l in range(0, len(list)):
        sum = sum + list[l]
    return sum / len(list)


for i in range(0, 20):
    print("Running NO.:" + str(i + 1))

    train_X = np.load('./data.npy')
    train_Y = train_X

    n_types = 3*NUM
    n_CV = 0


    option = op(N=2000, L=8, C=2048, scale=1, seed=0, nCV=0, n_types=n_types)
    # N_range = [256, 512, 1024]
    # N_range = [64, 128, 256, 512]
    # N_range = [16, 32, 64]
    N_range = [2000]
    # L = 32
    L = 8
    option.scale = 1
    # C_range = np.append(0,2.**np.arange(-6, 12, 2))
    # C_range = 2.**np.arange(-6, 12, 2)
    C_range = [2048]

    Models_tmp = []
    Models = []

    train_acc_result = np.zeros((n_CV, 1))
    train_time_result = np.zeros((n_CV, 1))

    MAX_acc = 0
    option_best = op(N=2000, L=8, C=2048, scale=1, seed=0, nCV=0, n_types=n_types)

    st = time.time()
    for n in N_range:
        option.N = n
        for j in C_range:
            option.C = j
            for l in range(1, L):
                sto = time.time()
                option.L = l
                [model_tmp, training_time_temp, training_loss_temp] = RVFLv(train_X, train_Y, option)
                print('Training Time for one option set:{:.2f}'.format(time.time() - sto))
                if time.time() - sto > 10:
                    print('current settings:{}'.format(option.__dict__))
    
    [model_RVFLv, train_time0, training_loss0] = RVFLv(train_X, train_Y, option_best)
    print('Training Time for one fold set:{:.2f}'.format(time.time() - st))
    Models.append(model_RVFLv)

    # train_acc_list.append(train_acc0)
    train_time_list.append(train_time0)
    train_loss_list.append(training_loss0)
    del model_RVFLv
    print('Best Train accuracy in:{}'.format(train_acc_result))
    mean_train_acc = train_acc_result.mean()
    print('Train accuracy:{}'.format(train_acc_result))
    print('Mean train accuracy:{}'.format(mean_train_acc))
    print("-----------------------------------------------------------------------------------")

print("-----------------------------------------------------------------------------------")
print("train_acc_avg:", get_avg(train_acc_list))
print("train_loss_avg:", get_avg(train_loss_list))
print("train_time_avg:", get_avg(train_time_list))
