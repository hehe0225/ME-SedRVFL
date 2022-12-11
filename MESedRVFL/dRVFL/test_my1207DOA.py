import numpy as np
from RVFLNN import RVFLNN
import matplotlib.pyplot as plt
import datetime
import scipy.io as sio

N = 21  # array number
n = np.arange(0, N, 1).reshape([-1, 1])
# path_save_net_pic = './net_pic.jpg'
phi = 32
f = 500
c = 1500
lmbda = c / f  
k=2*np.pi*f/c  
Lr=np.linspace(0,2,N)  

def sin_generation(size=10):
    X = np.random.uniform(low=-6.28, high=6.28, size=size).reshape(-1,1)
    Y = np.sin(X).reshape(1,-1)
    return X, Y

def data_generation(theta=60,SNR=0):
    
    x=Lr*k*np.cos(theta*np.pi/180)
    
    Pr1=np.cos(x)
    Pr1_i=np.sin(x)

    signal = np.hstack((Pr1,Pr1_i))
    #random.shuffle(signal)
    x=np.reshape(signal,(1,2*N))
    
    noise = np.random.randn(1,2*N) 	
    noise = noise-np.mean(noise) 								
    signal_power = np.linalg.norm( x )**2 / x.size	
    noise_variance = signal_power/np.power(10,(SNR/10))         
    noise = (np.sqrt(noise_variance) / np.std(noise) )*noise   
    signal_noise = noise + x
    x = signal_noise[0]
    
    x=np.reshape(x,(1,2*N))
    
    # lengthData=6000
    # test = './Dat.mat'
    # Test=sio.loadmat(test)
    # Test_Input=np.array(Test['Dat']).astype(np.complex)
    # x = Test_Input.T
    # x_r = np.real(x)
    # x_i = np.imag(x)
    # x = np.concatenate((x_r, x_i), axis=1)
    # # x=np.reshape(x,(700,512))
    # x=np.reshape(x,(lengthData,64)).T
    
    x_dnn=(x[0,0:N]+x[0,N:2*N])/np.max(x[0,0:N]+x[0,N:2*N])
    # plt.plot(Lr,x_dnn,c='red',label='true')
    y_true = x_dnn
    return x_dnn,y_true

font = {
#   'family' : 'Times New Roman',
   'weight' : 'normal',
   'size' : 18,
}

def visualization(X_train,y_train,X_test,y_test,pred_train,pred_test):
    # plt.plot(X_train, y_train, color=(70/255,130/255,180/255),label='Train',lw = '2')
    # plt.plot(X_test, y_test, linestyle='--', color='#FF69B4',label='Test')
    # plt.scatter(X_test, pred_test, color='g', label='Prediction(Test)')
    # plt.scatter(X_train, pred_train, c='none',marker='o',edgecolors='c', label='Prediction(Train)')
    
    plt.plot(Lr, y_train,color='r',label='true')
    plt.plot(Lr, y_test, color='b', linestyle='--',marker='>',label='predict')
    np.save('ME-SELM_10m_res_true.npy',y_train)
    np.save('ME-SELM_10m_res_pred.npy',y_test)
    
    plt.xlabel('Array length (m)',font)
    plt.ylabel('Amplitude (Normalized)',font)
    plt.rcParams.update({'font.size': 12})
    plt.legend(loc='upper right')
    plt.show()


if __name__=='__main__':
    # configuration
    # file_path1='./train1.mat'
    # file_path2='./Y_train1.mat'
    # file_path3='./test1.mat'
    # file_path4='./Y_test1.mat'
    # data_train=sio.loadmat(file_path1)
    # data_train_Y=sio.loadmat(file_path2)
    # data_test=sio.loadmat(file_path3)
    # data_test_Y=sio.loadmat(file_path4)
    # X_train=np.array(data_train['train1'])
    # y_train=np.array(data_train_Y['Y_train1'])
    # X_test=np.array(data_test['test1'])
    # y_test=np.array(data_test_Y['Y_test1'])
    
    # train_label=[]
    # test_label=[]
    # for i in range(y_train.shape[0]):
    #     for j in range(y_train.shape[1]):
    #         if(y_train[i,j]==1):
    #             train_label.append(j)
    # train_label=np.array(train_label, dtype = int)
    
    # for i in range(y_test.shape[0]):
    #     for j in range(y_test.shape[1]):
    #         if(y_test[i,j]==1):
    #             test_label.append(j)
    # test_label=np.array(test_label, dtype = int)
    
    
    # train_size= 100
    # test_size = 80
    epoch = 12300 # 12300
    time = datetime.datetime.now()
    # generate samples of sin function
    # X_train, y_train = sin_generation(size=train_size)
    # X_test, y_test = sin_generation(size=test_size)
    X_train, y_train = data_generation(theta=32,SNR=30)
    X_test, y_test = data_generation(theta=32,SNR=30)
    # train
    net = RVFLNN(1,500)
    net.train(X_train,y_train,epoch=epoch)
    prediction, error = net.predict(X_test, y_test)
    error = np.sum(np.abs(error))/len(X_test)
    # print results
    print("Epoch={}\nError={}\nTime={}".format(epoch,error,datetime.datetime.now()-time))
    pred_train,_= net.predict(X_train, y_train)
    visualization(X_train,y_train,X_test,y_test,pred_train,prediction)


from scipy import spatial

y=np.zeros([2*N,])
err=[]
for i in range(180):
    xx=Lr*k*np.cos(i*np.pi/180)
    y=np.sin(xx)
    y=y.tolist()
    cos_sim = 1 - spatial.distance.cosine(y, prediction[0])
    err.append(cos_sim)


print(err.index(max(err)))