import numpy as np
np.random.seed(167)
import scipy.io as sio
from keras.models import Sequential,Model,load_model
from keras.layers import Dense,Activation,Dropout
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import math
import keras
import random
import tensorflow as tf
from sklearn import preprocessing
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K


global NUM

d_iter=0
nb_epoch = 1000
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

#train='./data/snr=-5/data.mat'
#Train=sio.loadmat(train)
#Train_Input=np.array(Train['A']).astype(np.complex)
#Train_Input = np.transpose(Train_Input,(1,0))
#r = np.real(Train_Input)
#img = np.imag(Train_Input)
#signal = np.hstack((r,img))
#x_dnn=np.reshape(signal,(1,10))


# # add AWGN
# SNR = 0
# noise = np.random.randn(1,2*N) 	
# noise = noise-np.mean(noise) 								
# signal_power = np.linalg.norm( x_dnn )**2 / x_dnn.size	
# noise_variance = signal_power/np.power(10,(SNR/10))       
# noise = (np.sqrt(noise_variance) / np.std(noise) )*noise    
# signal_noise = noise + x_dnn
# x_dnn = signal_noise

#X_train_minmax = preprocessing.minmax_scale(x, feature_range=(0, 1), axis=1, copy=True)
#x_dnn = X_train_minmax

# snap=17
# loaddata='./data_paper/jasa/f_dnn.mat'  
# Train=sio.loadmat(loaddata)
# x=np.array(Train['f_dnn'])
# real = np.real(x)
# image = np.imag(x)
# x = np.concatenate((real, image), axis=1)
# x_dnn = x

def my_relu(x):
    return K.relu(x)**2-3*K.relu(x-1)**2+3*K.relu(x-2)**2-K.relu(x-3)**2

get_custom_objects().update({'my_relu':Activation(my_relu)})
 

model = Sequential()
model.add(Dense(units=1024, activation='tanh', input_shape=(2*N,)))
for layer in model.layers:
    layer.trainable = False
    print(layer.trainable)
model.add(Dropout(0.4))
model.add(Dense(units=512, activation='tanh'))
model.add(Dense(units=256, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(units=128, activation='tanh'))
model.add(Dense(units=64, activation='tanh'))
model.add(Dense(units=32, activation='tanh'))
model.add(Dense(units=16, activation='tanh'))
R=model.add(Dense(units=3*NUM,activation='sigmoid'))

def loss1(origin,r):
    xr=tf.zeros([N,])
    xr_i=tf.zeros([N,])
    for n in range(NUM):
        xr=xr+r[0,n]*tf.cos(k*tf.cos(180*r[0,n+NUM]*math.pi/180)*Lr+2*math.pi*r[0,n+2*NUM])
        xr_i=xr_i+r[0,n]*tf.sin(k*tf.cos(180*r[0,n+NUM]*math.pi/180)*Lr+2*math.pi*r[0,n+2*NUM])
    loss=(tf.norm(origin[0][0:N]-xr,ord=2)+tf.norm(origin[0][N:2*N]-xr_i,ord=2))/10+0.0*tf.norm(r[0,0:(NUM-1)],ord=1)
    return loss

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        if loss_type == 'epoch':
            plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('loss')
        plt.legend(loc="upper right")

def mse(origin,r):
    l2=tf.norm(r[0,0:(NUM-1)],ord=1)
    # MSE=np.sum(np.power((1 - r[0,NUM:2*NUM]),2))/len(origin)
    return l2


history = LossHistory()

def main():
    
    model.compile(loss=loss1,optimizer='adam',metrics=[mse])
    model.summary()
    dnn_err=np.zeros([2,])
     # plot_model(model, to_file=path_save_net_pic, show_shapes=True)
    model.fit(x_dnn, x_dnn, epochs=nb_epoch, batch_size=32,callbacks=[history])
    # plt.figure()
    # history.loss_plot('epoch')
    R=(model.predict(x_dnn))
    #------------------------------------------------------------------------------ save model ------------------------------------------------------------------------------
    # print("Saving model to disk \n")
    # mp = "./bf_saveModels/re_model.h5"
    # model.save(mp)
    
    #------------------------------------------------------------------------------ load model ------------------------------------------------------------------------------
    #print("Using saved model to predict...")
    #weight_path = './bf_saveModels/re_model.h5'
    #model = load_model(weight_path,custom_objects={'loss1': loss1,'mea':mea})
    #R = (model.predict(x))
    theta=np.zeros([180,1])
    spectrum=np.zeros([180,1])
    for m in range(180):
        theta[m,0]=1*m
    sigma=0.005
    for m in range(180):
        for n in range(NUM):
            spectrum[m,0]=spectrum[m,0]+R[0,n]*math.exp(-(np.cos(theta[m,0]*np.pi/180)-np.cos(np.pi*R[0,n+NUM]))*(np.cos(theta[m,0]*np.pi/180)-np.cos(np.pi*R[0,n+NUM]))/2/sigma/sigma)
    y_dnn = np.abs(spectrum)/np.max(abs(spectrum))
    y_dnn = y_dnn.tolist()
    y_dnn_index = y_dnn.index(max(y_dnn))


    #------------------------------------------------------------------------------ reconstruction ------------------------------------------------------------------------------
    y=np.zeros([N,])
    for n in range(NUM):
        y=y+R[0,n]*(np.cos(k*np.cos(R[0,n+NUM]*np.pi)*Lr+2*math.pi*R[0,n+2*NUM])+np.sin(k*np.cos(R[0,n+NUM]*np.pi)*Lr+2*math.pi*R[0,n+2*NUM]))

    
    #------------------------------------------------------------------------------ error ------------------------------------------------------------------------------
    print('DNN_error1:',theta[y_dnn_index],phi,theta[y_dnn_index]-phi)
    #------------------------------------------------------------------------------ plot ------------------------------------------------------------------------------

    # plt.figure()
    # plt.plot(theta+1,(np.abs(spectrum)/np.max(abs(spectrum))),label="dnn",c='blue')
    
    # plt.figure()
    # plt.plot(Lr,(x_dnn[0,0:N]+x_dnn[0,N:2*N])/np.max(x_dnn[0,0:N]+x_dnn[0,N:2*N]),c='red',label='true')
    # plt.plot(Lr,y/np.max(y),c='blue',label='predict',linestyle=':')
    # plt.legend()
    # plt.show()

if __name__ == '__main__':
    if d_iter<4:
        main()
    d_iter+=1


