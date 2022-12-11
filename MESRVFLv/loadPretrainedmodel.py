import numpy as np
np.random.seed(167)
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
nb_epoch = 1000
NUM= 2      # maximum possible target numbers
f = 500
c = 1500
lmbda = c / f  # wavelength
k=2*np.pi*f/c   # wavenumber
phi1 = 60
phi2= 65    # azimuth
phi3= 100   
phi4= 130  
phi5= 150   
Lr=np.linspace(0,3,21)  #array length and interval

#
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
signal = np.hstack((Pr1+Pr2,Pr1_i+Pr2_i))
#signal = np.hstack((Pr1,Pr1_i))
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

# add noise
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

# custom loss function
def loss1(origin,r):
    xr=tf.zeros([21,])
    xr_i=tf.zeros([21,])
    for n in range(NUM):
        xr=xr+r[0,n]*tf.cos(k*tf.cos(180*r[0,n+NUM]*math.pi/180)*Lr+2*math.pi*r[0,n+2*NUM])
        xr_i=xr_i+r[0,n]*tf.sin(k*tf.cos(180*r[0,n+NUM]*math.pi/180)*Lr+2*math.pi*r[0,n+2*NUM])
    loss=(tf.norm(origin[0][0:21]-xr,ord=2)+tf.norm(origin[0][21:42]-xr_i,ord=2))/10+0.008*tf.norm(r[0,0:(NUM-1)],ord=1)
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

# model.compile(loss=loss1,optimizer='adam',metrics=[mea])
# model.summary()
# model.fit(x, x, epochs=nb_epoch, batch_size=32,callbacks=[history])
# plt.figure()
# history.loss_plot('epoch')

#------------------------------------------------------------------------------ save model ------------------------------------------------------------------------------

#print("Saving model to disk \n")
#mp = "./bf_Multi_saveModels/multi_2_snr=15.h5"
#model.save(mp)
##------------------------------------------------------------------------------ load model ------------------------------------------------------------------------------
#print("Using saved model to predict...")
#weight_path = './bf_Multi_saveModels/multi_2_snr=15.h5'
#model = load_model(weight_path,custom_objects={'loss1': loss1,z'mea':mea})
R = (model.predict(x))


#---------------------------------------------------------------------------- retrain --------------------------------------------------------------------------------
# model.fit(x, x,batch_size=32,epochs=60000)
## test data for evaluation
# loss,accuracy = model.evaluate(x_test,x_test)
# print('\ntest loss',loss)
# print('accuracy',accuracy)
## save and load
# model.save_weights('./bf_Multi_saveModels/model_1_weights.h5')
# model.load_weights('./bf_Multi_saveModels/model_1_weights.h5')

# from keras.models import model_from_json
# json_string = model.to_json()
# model = model_from_json(json_string)
# R = (model.predict(x))
# print(json_string)

#-------------------------------------------------------------------------------plot---------------------------------------------------------------------
y=np.zeros([21,])
for n in range(NUM):
    y=y+R[0,n]*(np.cos(k*np.cos(R[0,n+NUM]*np.pi)*Lr+2*math.pi*R[0,n+2*NUM])+np.sin(k*np.cos(R[0,n+NUM]*np.pi)*Lr+2*math.pi*R[0,n+2*NUM]))


theta=np.zeros([180,1])
spectrum=np.zeros([180,1])
for m in range(180):
    theta[m,0]=1*m
sigma=0.005
for m in range(180):
    for n in range(NUM):
        spectrum[m,0]=spectrum[m,0]+R[0,n]*math.exp(-(np.cos(theta[m,0]*np.pi/180)-np.cos(np.pi*R[0,n+NUM]))*(np.cos(theta[m,0]*np.pi/180)-np.cos(np.pi*R[0,n+NUM]))/2/sigma/sigma)

#np.save('dnn_2.npy',spectrum)


#----------------------------------------------CBF------------------------------------
k = 2*np.pi*f/c
Pr = np.exp(1j*Lr*k*np.cos(phi1*np.pi/180))
Pr2 = np.exp(1j*Lr*k*np.cos(phi2*np.pi/180))
Pr3 = np.exp(1j*Lr*k*np.cos(phi3*np.pi/180))
Pr4 = np.exp(1j*Lr*k*np.cos(phi4*np.pi/180))
Pr5 = np.exp(1j*Lr*k*np.cos(phi5*np.pi/180))
x_cbf=Pr+Pr2


theta=np.zeros([180,1])
spectrum=np.zeros([180,1])
for m in range(180):
    theta[m,0]=1*m
    
theta0=np.zeros([361,1])
data_music=np.zeros([361,1])
for m in range(361):
    theta0[m,0]=0.5*m

doa=np.zeros([len(theta),1])
for m in range(len(theta)):
    doa[m,0]=np.abs(np.sum(x_cbf*np.exp(-1j*Lr*k*np.cos(theta[m,0]*np.pi/180))))

#--------------------------------------------CBF-------------------------------------------

#-------------------------------------------MUSIC------------------------------------------
data_music='MUSIC.mat'
data_music=sio.loadmat(data_music)
data_music=np.array(data_music['Pmusic']).astype(np.float32)
data_music = np.transpose(data_music,(1,0))
#-------------------------------------------MUSIC------------------------------------------

#-------------------------------------------MVDR------------------------------------------
data_mvdr=np.load('MVDR.npy')
#-------------------------------------------MVDR------------------------------------------


plt.figure()
data_dnn=np.load('nn.npy')
data_dnn[92:130]=0
plt.plot(theta,np.abs(doa)/np.max(doa),label="CBF",linestyle=':')
plt.plot(theta0,np.abs(data_music)/np.max(np.abs(data_music)),label="MUSIC",marker='x',markersize=7)
plt.plot(theta+1,np.abs(data_dnn)/np.max(abs(data_dnn)),label="ME-SRVFLv",c='blue')
plt.plot(theta,np.abs(data_mvdr)/np.max(np.abs(data_mvdr)),label="MVDR",linestyle='--')
plt.plot([phi1,phi1],[0.91,0.91],c='red',marker='^',markersize=7,label='True')
plt.plot([phi2,phi2],[0.75,0.75],c='red',marker='^',markersize=7)

font = {
#   'family' : 'Times New Roman',
   'weight' : 'normal',
   'size' : 15,
}
plt.xlabel('Angle(°)',font)
plt.ylabel('Amplitude',font)
plt.legend()
plt.show()



plt.legend()
plt.show()



