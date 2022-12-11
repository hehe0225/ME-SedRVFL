import numpy as np
from dRVFLimport dRVFL
import matplotlib.pyplot as plt
import datetime
import scipy.io as sio

f = 500
c = 1500
lmbda = c / f  # wavelength
k=2*np.pi*f/c   # beam number
# phi1 = 10
# phi2= 20    
# phi3= 35    
# phi4= 45   
# phi5= 60    
N=64 # array number
Lr=np.linspace(0,10,N)  # array 10m

def sin_generation(size=10):
    X = np.random.uniform(low=-6.28, high=6.28, size=size).reshape(-1,1)
    Y = np.sin(X).reshape(1,-1)
    return X, Y

def data_generation(theta=60,SNR=0):
    
    x=Lr*k*np.cos(theta*np.pi/180)
    
    noise = np.random.randn(1,N) 	
    noise = noise-np.mean(noise) 								
    signal_power = np.linalg.norm( x )**2 / x.size
    noise_variance = signal_power/np.power(10,(SNR/10))         
    noise = (np.sqrt(noise_variance) / np.std(noise) )*noise    
    signal_noise = noise + x
    x = signal_noise[0]
    
    Pr1=np.cos(x)
    Pr1_i=np.sin(x)

    
    signal = np.hstack((Pr1_i))
    
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
    
    
    y_true = signal
    return x,y_true

font = {
#   'family' : 'Times New Roman',
   'weight' : 'normal',
   'size' : 15,
}

def visualization(X_train,y_train,X_test,y_test,pred_train,pred_test):
    plt.plot(X_train, y_train, color=(70/255,130/255,180/255),label='Train',lw = '2')
    plt.plot(X_test, y_test, linestyle='--', color='#FF69B4',label='Test')
    plt.scatter(X_test, pred_test, color='g', label='Prediction(Test)')
    plt.scatter(X_train, pred_train, c='none',marker='o',edgecolors='c', label='Prediction(Train)')
    
    
    plt.xlabel('x',font)
    plt.ylabel('y',font)
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
    X_train, y_train = data_generation(theta=45,SNR=30)
    X_test, y_test = data_generation(theta=30,SNR=30)
    # train
    net = dRVFL(1,50)
    net.train(X_train,y_train,epoch=epoch)
    prediction, error = net.predict(X_test, y_test)
    error = np.sum(np.abs(error))/len(X_test)
    # print results
    print("Epoch={}\nError={}\nTime={}".format(epoch,error,datetime.datetime.now()-time))
    pred_train,_= net.predict(X_train, y_train)
    visualization(X_train,y_train,X_test,y_test,pred_train,prediction)


from scipy import spatial

y=np.zeros([64,])
err=[]
for i in range(180):
    xx=Lr*k*np.cos(i*np.pi/180)
    y=np.sin(xx)
    y=y.tolist()
    cos_sim = 1 - spatial.distance.cosine(y, prediction[0])
    err.append(cos_sim)


print(err.index(max(err)))