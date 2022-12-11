import sys
import numpy as np
import sklearn.metrics
from sklearn.metrics import log_loss
import time
from Logger import Logger
import scipy.io as sio
import warnings
warnings.filterwarnings("ignore")


class RVFL:
    
    def __init__(self, n_nodes, lam, w_random_vec_range, b_random_vec_range, activation, same_feature=False,
                 task_type='classification'):
        assert task_type in ['classification', 'regression'], 'task_type should be "classification" or "regression".'
        self.n_nodes = n_nodes
        self.lam = lam
        self.w_random_range = w_random_vec_range
        self.b_random_range = b_random_vec_range
        self.random_weights = None
        self.random_bias = None
        self.beta = None
        a = Activation()
        self.activation_function = getattr(a, activation)
        self.data_std = None
        self.data_mean = None
        self.same_feature = same_feature
        self.task_type = task_type

    def train(self, data, label):
        """

        :param data: Training data.
        :param label: Training label.
        :param n_class: An integer of number of class. In regression, this parameter won't be used.
        :return: No return
        """

        assert len(data.shape) > 1, 'Data shape should be [n, dim].'
        assert len(data) == len(label), 'Label number does not match data number.'
        # assert len(label.shape) == 1, 'Label should be 1-D array.'
        data = self.standardize(data)  # Normalization data
        n_sample = len(data)
        n_feature = len(data[0])
        st = time.time()

        self.random_weights = self.get_random_vectors(n_feature, self.n_nodes, self.w_random_range)
        self.random_bias = self.get_random_vectors(1, self.n_nodes, self.b_random_range)
        #h = self.activation_function(np.dot(data, self.random_weights) + np.dot(np.ones([n_sample, 1]), self.random_bias))
        h = np.dot(data, self.random_weights) + np.dot(np.ones([n_sample, 1]), self.random_bias)
        h, k, b = self.Scale(h, scale_ratio, scale_mode)
        h = self.activation_function(h*k+b)
        d = np.concatenate([h, data], axis=1)
        d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
        y = label

        if n_sample > (self.n_nodes + n_feature):
            self.beta = np.linalg.inv((self.lam * np.identity(d.shape[1]) + np.dot(d.T, d))).dot(d.T).dot(y)
        else:
            self.beta = d.T.dot(np.linalg.inv(self.lam * np.identity(n_sample) + np.dot(d, d.T))).dot(y)
        # print(self.beta)
        output = np.dot(d, self.beta)
        train_time = time.time() - st
        if self.task_type == 'classification':
            proba = self.softmax(output)
            result = np.argmax(proba, axis=1)
            result1 = self.one_hot(result, n_class)
            crossentropy = log_loss(label, result1)
            label = np.argmax(label, axis=1)
            acc = np.sum(np.equal(result, label)) / len(result)
            f1_score=sklearn.metrics.f1_score(label, result, average='weighted')
        return acc, crossentropy, f1_score,train_time

    def predict(self, data):
        """

        :param data: Predict data.
        :return: When classification, return Prediction result and probability.
                 When regression, return the output of rvfl.
        """
        data = self.standardize(data)  # Normalization data
        #h = self.activation_function(np.dot(data, self.random_weights) + self.random_bias)
        h = np.dot(data, self.random_weights) + self.random_bias
        h, k, b = self.Scale(h, scale_ratio, scale_mode)
        h = self.activation_function(h * k + b)
        d = np.concatenate([h, data], axis=1)
        d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
        output = np.dot(d, self.beta)
        if self.task_type == 'classification':
            proba = self.softmax(output)
            result = np.argmax(proba, axis=1)
            return result, proba
        elif self.task_type == 'regression':
            return output

    def eval(self, result, label):
        """
        :param result: Prediction result.
        :param data: Evaluation data.
        :param label: Evaluation label.
        :return: When classification return accuracy.
                 When regression return MAE.
        """
        if self.task_type == 'classification':
            # result = np.argmax(output, axis=1)
            result1 = self.one_hot(result, n_class)
            crossentropy = log_loss(label, result1)
            label = np.argmax(label, axis=1)
            acc = np.sum(np.equal(result, label)) / len(result)
            f1_score = sklearn.metrics.f1_score(label, result, average='weighted')
            return acc, crossentropy, f1_score

        elif self.task_type == 'regression':
            mae = np.mean(np.abs(result - label))
            mape = mae/np.mean(label)
            return mae, mape

    def Scale(self, H, scale_ratio, scale_mode):
        Saturating_threshold = np.array([-50, 50])
        #Saturating_threshold_activate = np.array([0, 1])
        if scale_mode==1:
            [H, k, b] = self.Scale_feature(H, Saturating_threshold, scale_ratio)
        elif scale_mode==2:
            [H, k, b] = self.Scale_feature_separately(H, Saturating_threshold, scale_ratio)
        return H, k, b

    def Scale_feature(self, Input, Saturating_threshold, ratio):
        Min_value = Input.min()
        Max_value = Input.max()
        min_value = Saturating_threshold[0] * ratio
        max_value = Saturating_threshold[1] * ratio
        k = (max_value - min_value) / (Max_value - Min_value)
        b = (min_value * Max_value - Min_value * max_value) / (Max_value - Min_value)
        Output = Input * k + b
        return Output, k, b

    def Scale_feature_separately(self, Input, Saturating_threshold, ratio):
        nNeurons = Input.shape[1]
        k = np.zeros((1, nNeurons))
        b = np.zeros((1, nNeurons))
        Output = np.zeros(Input.shape)
        min_value = Saturating_threshold[0] * ratio
        max_value = Saturating_threshold[1] * ratio
        for i in range(0, nNeurons):
            Min_value = np.min(Input[:, i])
            Max_value = np.max(Input[:, i])
            k[0, i] = (max_value - min_value) / (Max_value - Min_value)
            b[0, i] = (min_value * Max_value - Min_value * max_value) / (Max_value - Min_value)
            Output[:, i] = Input[:, i] * k[0, i] + b[0, i]
        return Output, k, b


    @staticmethod
    def get_random_vectors(m, n, scale_range):
        x = (scale_range[1] - scale_range[0]) * np.random.random([m, n]) + scale_range[0]
        return x

    @staticmethod
    def one_hot(x, n_class):
        y = np.zeros([len(x), n_class])
        for i in range(len(x)):
            y[i, x[i]] = 1
        return y


    def standardize(self, x):
        if self.same_feature is True:
            if self.data_std is None:
                self.data_std = np.maximum(np.std(x), 1/np.sqrt(len(x)))
            if self.data_mean is None:
                self.data_mean = np.mean(x)
            return (x - self.data_mean) / self.data_std
        else:
            if self.data_std is None:
                self.data_std = np.maximum(np.std(x, axis=0), 1/np.sqrt(len(x)))
            if self.data_mean is None:
                self.data_mean = np.mean(x, axis=0)
            return (x - self.data_mean) / self.data_std


    @staticmethod
    def softmax(x):
        return np.exp(x) / np.repeat((np.sum(np.exp(x), axis=1))[:, np.newaxis], len(x[0]), axis=1)


class Activation:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.e ** (-x))

    @staticmethod
    def sine(x):
        return np.sin(x)

    @staticmethod
    def hardlim(x):
        return (np.sign(x) + 1) / 2

    @staticmethod
    def tribas(x):
        return np.maximum(1 - np.abs(x), 0)

    @staticmethod
    def radbas(x):
        return np.exp(-(x**2))

    @staticmethod
    def sign(x):
        return np.sign(x)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)



if __name__ == '__main__':
    sys.stdout = Logger(file_path="/result")

    def prepare_data_classify4(proportion):
        ## import the dataset4
        file_path = '../../dataset/train_5403.mat'
        data1 = sio.loadmat(file_path)
        # np.set_printoptions(threshold=np.inf)
        data2 = data1['sigXY']
        # print(data.shape)
        data = data2[:, :16]
        label = data2[:, 16:]
        shuffle_index = np.arange(len(data2))
        np.random.shuffle(shuffle_index)

        train_number = int(proportion * len(data2))
        train_index = shuffle_index[:train_number]
        val_index = shuffle_index[train_number:]

        data_train = data[train_index]
        label_train = label[train_index]
        # print(label_train)
        data_val = data[val_index]
        label_val = label[val_index]

        print("label_train.shape:" + str(label_train.shape))
        print("label_val.shape:" + str(label_val.shape))

        return (data_train, label_train), (data_val, label_val), len(label_val[0])

    def prepare_data_classify5(proportion):
        ## import dataset5
        file_path = '../../dataset/DATA2.mat'
        data1 = sio.loadmat(file_path)
        data2 = data1['sigXY']
        data = data2[:, :16]
        label = data2[:, 16:]
        shuffle_index = np.arange(len(data2))
        np.random.shuffle(shuffle_index)

        train_number = int(proportion * len(data2))
        train_index = shuffle_index[:train_number]
        val_index = shuffle_index[train_number:]

        data_train = data[train_index]
        label_train = label[train_index]
        # print(label_train)
        data_val = data[val_index]
        label_val = label[val_index]
        print("label_train.shape:" + str(label_train.shape))
        print("label_val.shape:" + str(label_val.shape))
        return (data_train, label_train), (data_val, label_val), len(label_val[0])


    def get_avg(list):
        sum=0
        for l in range(0,len(list)):
            sum=sum+list[l]
        return sum/len(list)

    # Classification
    #num_nodes = 500  # Number of enhancement nodes.
    #num_nodes = 1000
    #num_nodes = 1500
    num_nodes = 2000
    regular_para = 2  # Regularization parameter.
    weight_random_range = [-1, 1]  # Range of random weights.
    bias_random_range = [0, 1]  # Range of random bias.
    scale_ratio = 0.9
    scale_mode = 1

    train, val, n_class = prepare_data_classify5(0.7)
    # train, val，n_class = prepare_data_classify4(0.7)
    train_acc_list = []
    pred_acc_list = []
    train_loss_list = []
    pred_loss_list = []
    train_time_list=[]

    print("num_nodes:", num_nodes)
    print("regular_para:", regular_para)
    print("weight_random_range:", weight_random_range)
    print("bias_random_range:", bias_random_range)
    print("scale_ratio:", scale_ratio)
    print("scale_mode:", scale_mode)
    print("------------------------------------------------------")

    for i in range(0, 20):
        print("Running NO.:" + str(i + 1))
        rvfl = RVFL(n_nodes=num_nodes, lam=regular_para, w_random_vec_range=weight_random_range,
                    b_random_vec_range=bias_random_range, activation='relu', same_feature=False,
                    task_type='classification')
        #st = time.time()
        train_accuracy, cross_entropy,train_f1_score,train_time = rvfl.train(train[0], train[1])
        #train_time = time.time() - st
        print('Training Time:{:.2f}'.format(train_time))
        train_time_list.append(train_time)
        print('train_accuracy:', train_accuracy)
        train_acc_list.append(train_accuracy)
        print('train_cross_entropy:', cross_entropy)
        train_loss_list.append(cross_entropy)
        print('train_f1_score:', train_f1_score)
        prediction, proba = rvfl.predict(val[0])
        # print('prediction:', prediction)
        # print(proba)
        pred_accuracy, pred_cross_entropy, pred_f1_score = rvfl.eval(prediction, val[1])
        print('pred_accuracy:', pred_accuracy)
        pred_acc_list.append(pred_accuracy)
        print('pred_cross_entropy:', pred_cross_entropy)
        pred_loss_list.append(pred_cross_entropy)
        print('pred_f1_score:', pred_f1_score)
        print("-----------------------------------------------------")
    print("-----------------------------------------------------")
    print("train_acc_avg:", get_avg(train_acc_list))
    print("pred_acc_avg:", get_avg(pred_acc_list))
    print("train_loss_avg:", get_avg(train_loss_list))
    print("pred_loss_avg:", get_avg(pred_loss_list))
    print("train_time_avg:", get_avg(train_time_list))

