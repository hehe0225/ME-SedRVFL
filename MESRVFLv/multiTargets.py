import numpy as np
# np.random.seed(167)
import scipy.io as sio
from keras.models import Sequential,Model,load_model
from keras.layers import Dense,Activation
# from keras.optimizers import SGD
# from keras.utils import plot_model
import matplotlib.pyplot as plt
import math
import keras
import tensorflow as tf
from sklearn import preprocessing
global NUM
nb_epoch = 3000
NUM= 10      #考虑来自10个方位的目标
# path_save_net_pic = './net_pic.jpg'
f = 500
c = 1500
lmbda = c / f  #wavelength
k=2*np.pi*f/c   #wavenumber
phi1 = 10
phi2= 20    #azimuth
phi3= 35    
phi4= 45    
phi5= 60   
Lr=np.linspace(0,10,21)  #array length and interval


## Simulation data
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
x = signal


# load actual experiments underwater data
#import scipy.io as scio 
#matfn='./data/multi.mat'  
#Test=scio.loadmat(matfn)
#x=np.array(Test['x']).astype(np.complex)
#x = np.transpose(x,(1,0))
#real = np.real(x)
#image = np.imag(x)
#x = np.concatenate((real, image), axis=0)

#  add noise
#SNR = 30
#SNR = 10
#noise = np.random.randn(1,42) 
#noise = noise-np.mean(noise) 								
#signal_power = np.linalg.norm( x )**2 / x.size	
#noise_variance = signal_power/np.power(10,(SNR/10))         
#noise = (np.sqrt(noise_variance) / np.std(noise) )*noise   
#signal_noise = noise + x
#x = signal_noise

x=np.reshape(x,(1,42))
model = Sequential()
model.add(Dense(units=1024, activation='tanh', input_shape=(42,)))
model.add(Dense(units=512, activation='tanh'))
model.add(Dense(units=256, activation='tanh'))
model.add(Dense(units=256, activation='tanh'))
model.add(Dense(units=128, activation='tanh'))
model.add(Dense(units=128, activation='tanh'))
model.add(Dense(units=64, activation='tanh'))
model.add(Dense(units=64, activation='tanh'))
model.add(Dense(units=32, activation='tanh'))
model.add(Dense(units=16, activation='tanh'))
model.add(Dense(units=3*NUM,activation='sigmoid'))


# custom loss function (signal model based loss function)
def loss1(origin,r):
    xr=tf.zeros([21,])
    xr_i=tf.zeros([21,])
    for n in range(NUM):
        xr=xr+r[0,n]*tf.cos(k*tf.cos(180*r[0,n+NUM]*math.pi/180)*Lr+2*math.pi*r[0,n+2*NUM])
        xr_i=xr_i+r[0,n]*tf.sin(k*tf.cos(180*r[0,n+NUM]*math.pi/180)*Lr+2*math.pi*r[0,n+2*NUM])
    loss=(tf.norm(origin[0][0:21]-xr,ord=2)+tf.norm(origin[0][21:42]-xr_i,ord=2))/10+0.001*tf.norm(r[0,0:(NUM-1)],ord=1)
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
        # acc
        # plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        # plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            # plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            # plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
            plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('loss')
        plt.legend(loc="upper right")

def mea(origin,r):
    l1=0.02*tf.norm(r[0,0:(NUM-1)],ord=1)
    return l1

history = LossHistory()
learning_rate = 0.001
# sgd = SGD(lr=learning_rate, decay=learning_rate/nb_epoch, momentum=0.9, nesterov=True)

#------------------------------------------------------------------------------ train model ------------------------------------------------------------------------------

model.compile(loss=loss1,optimizer='adam',metrics=[mea])
model.summary()
model.fit(x, x, epochs=nb_epoch, batch_size=32,callbacks=[history])
plt.figure()
history.loss_plot('epoch')

#------------------------------------------------------------------------------ save model ------------------------------------------------------------------------------
#print("Saving model to disk \n")
#mp = "./bf_Multi_saveModels/multi_2_snr=15.h5"
#model.save(mp)
##------------------------------------------------------------------------------  load model ------------------------------------------------------------------------------
#print("Using saved model to predict...")
#weight_path = './bf_Multi_saveModels/multi_2_snr=15.h5'
#model = load_model(weight_path,custom_objects={'loss1': loss1,z'mea':mea})
R = (model.predict(x))


#----------------------------------------------------------------------------retrain --------------------------------------------------------------------------------
# model.fit(x, x,batch_size=32,epochs=60000)
## test data for evaluation
# loss,accuracy = model.evaluate(x_test,x_test)
# print('\ntest loss',loss)
# print('accuracy',accuracy)

## save and load
# model.save_weights('./bf_Multi_saveModels/model_1_weights.h5')
# model.load_weights('./bf_Multi_saveModels/model_1_weights.h5')

## save and load
# from keras.models import model_from_json
# json_string = model.to_json()
# model = model_from_json(json_string)
# R = (model.predict(x))
# print(json_string)

#-------------------------------------------------------------------------------plot---------------------------------------------------------------------
y=np.zeros([21,])
for n in range(NUM):
    y=y+R[0,n]*(np.cos(k*np.cos(R[0,n+NUM]*np.pi)*Lr+2*math.pi*R[0,n+2*NUM])+np.sin(k*np.cos(R[0,n+NUM]*np.pi)*Lr+2*math.pi*R[0,n+2*NUM]))

#plt.figure()
#plt.plot(Lr,(train_signal[0][0:200]+train_signal[0][200:400])/np.max(train_signal[0][0:200]+train_signal[0][200:400]),c='red',label='true')
#plt.plot(Lr,y/np.max(y),c='blue',label='predict')
#plt.legend()

theta=np.zeros([180,1])
spectrum=np.zeros([180,1])
for m in range(180):
    theta[m,0]=1*m
sigma=0.005
for m in range(180):
    for n in range(NUM):
        spectrum[m,0]=spectrum[m,0]+R[0,n]*math.exp(-(np.cos(theta[m,0]*np.pi/180)-np.cos(np.pi*R[0,n+NUM]))*(np.cos(theta[m,0]*np.pi/180)-np.cos(np.pi*R[0,n+NUM]))/2/sigma/sigma)


plt.figure()
plt.plot(theta,np.abs(spectrum)/np.max(abs(spectrum)),label="dnn",c='blue')
plt.plot([phi1,phi1],[0,1],c='red',linestyle='--',label='true')
plt.plot([phi2,phi2],[0,1],c='red',linestyle='--')
plt.plot([phi3,phi3],[0,1],c='red',linestyle='--')
plt.plot([phi4,phi4],[0,1],c='red',linestyle='--')
plt.plot([phi5,phi5],[0,1],c='red',linestyle='--')


plt.xlabel('θ')
plt.ylabel('A')
plt.legend()
plt.show()




