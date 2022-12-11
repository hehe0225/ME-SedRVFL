import os
import numpy as np
from option import option as op
from MESRVFlv import *
import time
import scipy.io as sio
from Logger import Logger
import sys
np.seterr(divide='ignore', invalid='ignore')

nb_epoch = 3000
# path_save_net_pic = './net_pic.jpg'
f = 500
c = 1500
lmbda = c / f  #波长
k=2*np.pi*f/c   #波束
phi1 = 10
phi2= 20    #入射角度
phi3= 35    #入射角度
phi4= 45    #入射角度
phi5= 60    #入射角度

Pr1=np.cos(Lr*k*np.cos(phi1*np.pi/180))
Pr1_i=np.sin(Lr*k*np.cos(phi1*np.pi/180))
Pr2=np.cos(Lr*k*np.cos(phi2*np.pi/180))
Pr2_i=np.sin(Lr*k*np.cos(phi2*np.pi/180))
Pr3=np.cos(Lr*k*np.cos(phi3*np.pi/180))
Pr3_i=np.sin(Lr*k*np.cos(phi3*np.pi/180))
Pr4=np.cos(Lr*k*np.cos(phi4*np.pi/180))
Pr4_i=np.sin(Lr*k*np.cos(phi4*np.pi/180))
Pr5=np.cos(Lr*k*np.cos(phi5*np.pi/180))
Pr5_i=np.sin(Lr*k*np.cos(phi5*np.pi/180))
signal = np.hstack((Pr1+Pr2+Pr3+Pr4+Pr5,Pr1_i+Pr2_i+Pr3_i+Pr4_i+Pr5_i))
data = signal

train_acc_list = []
train_loss_list = []
train_time_list = []


def get_avg(list):
    sum = 0
    for l in range(0, len(list)):
        sum = sum + list[l]
    return sum / len(list)


for i in range(0, 20):
    print("Running NO.:" + str(i + 1))

    train_X = data
    train_Y = data

    n_types = len(train_Y[0])
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
                [model_tmp, train_acc_temp, training_time_temp, training_loss_temp] = MESRVFLv(train_X, train_Y, data, option)
                if train_acc_temp > MAX_acc:
                    MAX_acc = train_acc_temp
                    option_best.acc_train = train_acc_temp.max()
                    option_best.C = option.C
                    option_best.N = option.N
                    option_best.L = option.L
                    # option_best.nCV = i
                    print('Temp Best Option:{}'.format(option_best.__dict__))
                print('Training Time for one option set:{:.2f}'.format(time.time() - sto))
                if time.time() - sto > 10:
                    print('current settings:{}'.format(option.__dict__))
    [model_MESRVFLv, train_acc0, train_time0, training_loss0] = MESRVFLv(train_X, train_Y,data,option_best)
    print('Training Time for one fold set:{:.2f}'.format(time.time() - st))
    Models.append(model_MESRVFLv)

    train_acc_list.append(train_acc0)
    train_time_list.append(train_time0)
    train_loss_list.append(training_loss0)
    del model_MESRVFLv
    print('Best Train accuracy in:{}'.format(train_acc_result
                                                                       ))
    mean_train_acc = train_acc_result.mean()
    print('Train accuracy:{}'.format(train_acc_result))
    print('Mean train accuracy:{}'.format(mean_train_acc))
    print("-----------------------------------------------------------------------------------")

print("-----------------------------------------------------------------------------------")
print("train_acc_avg:", get_avg(train_acc_list))
print("train_loss_avg:", get_avg(train_loss_list))
print("train_time_avg:", get_avg(train_time_list))
