import numpy as np
import random
import datetime

class EnsembleDeepRVFL:
    def __init__(self, n_nodes, lam, w_random_vec_range, b_random_vec_range, activation, n_layer, same_feature=False,
                 task_type='classification'):
        assert task_type in ['classification', 'regression'], 'task_type should be "classification" or "regression".'
        self.n_nodes = n_nodes
        self.lam = lam
        self.w_random_range = w_random_vec_range
        self.b_random_range = b_random_vec_range
        self.random_weights = []
        self.random_bias = []
        self.beta = []
        a = Activation()
        self.activation_function = getattr(a, activation)
        self.n_layer = n_layer
        self.data_std = [None] * self.n_layer
        self.data_mean = [None] * self.n_layer
        self.same_feature = same_feature
        self.task_type = task_type

    def train(self, data, label, n_class):
        """

        :param data: Training data.
        :param label: Training label.
        :param n_class: An integer of number of class. In regression, this parameter won't be used.
        :return: No return
        """

        assert len(data.shape) > 1, 'Data shape should be [n, dim].'
        assert len(data) == len(label), 'Label number does not match data number.'
        assert len(label.shape) == 1, 'Label should be 1-D array.'

        n_sample = len(data)
        n_feature = len(data[0])
        h = data.copy()
        data = self.standardize(data, 0)
        if self.task_type == 'classification':
            y = self.one_hot(label, n_class)
        else:
            y = label
        for i in range(self.n_layer):
            h = self.standardize(h, i)  # Normalization data
            self.random_weights.append(self.get_random_vectors(len(h[0]), self.n_nodes, self.w_random_range))
            self.random_bias.append(self.get_random_vectors(1, self.n_nodes, self.b_random_range))
            h = self.activation_function(np.dot(h, self.random_weights[i]) + np.dot(np.ones([n_sample, 1]),
                                                                                    self.random_bias[i]))
            d = np.concatenate([h, data], axis=1)

            h = d

            d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)

            if n_sample > (self.n_nodes + n_feature):
                self.beta.append(np.linalg.inv((self.lam * np.identity(d.shape[1]) + np.dot(d.T, d))).dot(d.T).dot(y))
            else:
                self.beta.append(d.T.dot(np.linalg.inv(self.lam * np.identity(n_sample) + np.dot(d, d.T))).dot(y))

    def predict(self, data, output_prob=False):
        """

        :param data: Predict data.
        :return: When classification, return vote result,  addition result and probability.
                 When regression, return the mean output of edrvfl.
        """
        n_sample = len(data)
        h = data.copy()
        data = self.standardize(data, 0)  # Normalization data
        outputs = []
        for i in range(self.n_layer):
            h = self.standardize(h, i)  # Normalization data
            h = self.activation_function(np.dot(h, self.random_weights[i]) + np.dot(np.ones([n_sample, 1]),
                                                                                    self.random_bias[i]))
            d = np.concatenate([h, data], axis=1)

            h = d

            d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
            outputs.append(np.dot(d, self.beta[i]))
        if self.task_type == 'classification':
            vote_res = [np.argmax(item, axis=1) for item in outputs]
            vote_res = list(map(np.bincount, list(np.array(vote_res).transpose())))
            vote_res = np.array(list(map(np.argmax, vote_res)))

            add_proba = self.softmax(np.sum(outputs, axis=0))
            add_res = np.argmax(add_proba, axis=1)
            return vote_res, (add_res, add_proba)
        elif self.task_type == 'regression':
            return np.mean(outputs, axis=0)


    def eval(self, data, label):
        """

        :param data: Evaluation data.
        :param label: Evaluation label.
        :return: When classification return vote and addition accuracy.
                 When regression return MAE.
        """

        assert len(data.shape) > 1, 'Data shape should be [n, dim].'
        assert len(data) == len(label), 'Label number does not match data number.'
        assert len(label.shape) == 1, 'Label should be 1-D array.'

        n_sample = len(data)
        h = data.copy()
        data = self.standardize(data, 0)
        outputs = []
        for i in range(self.n_layer):
            h = self.standardize(h, i)  # Normalization data

            h = self.activation_function(np.dot(h, self.random_weights[i]) + np.dot(np.ones([n_sample, 1]),
                                                                                    self.random_bias[i]))
            d = np.concatenate([h, data], axis=1)

            h = d

            d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)

            outputs.append(np.dot(d, self.beta[i]))
        if self.task_type == 'classification':
            vote_res = [np.argmax(item, axis=1) for item in outputs]
            vote_res = list(map(np.bincount, list(np.array(vote_res).transpose())))
            vote_res = np.array(list(map(np.argmax, vote_res)))
            vote_acc = np.sum(np.equal(vote_res, label)) / len(label)

            add_proba = self.softmax(np.sum(outputs, axis=0))
            add_res = np.argmax(add_proba, axis=1)
            add_acc = np.sum(np.equal(add_res, label)) / len(label)

            return vote_acc, add_acc
        elif self.task_type == 'regression':
            pred = np.mean(outputs, axis=0)
            mae = np.mean(np.abs(pred - label))
            return mae

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

    def standardize(self, x, index):
        if self.same_feature is True:
            if self.data_std[index] is None:
                self.data_std[index] = np.maximum(np.std(x), 1/np.sqrt(len(x)))
            if self.data_mean[index] is None:
                self.data_mean[index] = np.mean(x)
            return (x - self.data_mean[index]) / self.data_std[index]
        else:
            if self.data_std[index] is None:
                self.data_std[index] = np.maximum(np.std(x, axis=0), 1/np.sqrt(len(x)))
            if self.data_mean[index] is None:
                self.data_mean[index] = np.mean(x, axis=0)
            return (x - self.data_mean[index]) / self.data_std[index]

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

    @staticmethod
    def leaky_relu(x):
        x[x >= 0] = x[x >= 0]
        x[x < 0] = x[x < 0] / 10.0
        return x


if __name__ == '__main__':
    import sklearn.datasets as sk_dataset

    def prepare_data_classify(proportion):
        dataset = sk_dataset.load_breast_cancer()
        label = dataset['target']
        data = dataset['data']
        n_class = len(dataset['target_names'])

        shuffle_index = np.arange(len(label))
        np.random.shuffle(shuffle_index)

        train_number = int(proportion * len(label))
        train_index = shuffle_index[:train_number]
        val_index = shuffle_index[train_number:]
        data_train = data[train_index]
        label_train = label[train_index]
        data_val = data[val_index]
        label_val = label[val_index]
        return (data_train, label_train), (data_val, label_val), n_class

    def prepare_data_regression(proportion):
        dataset = sk_dataset.load_diabetes()
        label = dataset['target']
        data = dataset['data']

        shuffle_index = np.arange(len(label))
        np.random.shuffle(shuffle_index)

        train_number = int(proportion * len(label))
        train_index = shuffle_index[:train_number]
        val_index = shuffle_index[train_number:]
        data_train = data[train_index]
        label_train = label[train_index]
        data_val = data[val_index]
        label_val = label[val_index]
        return (data_train, label_train), (data_val, label_val)

    def data_generation(proportion=0,SNR=0,n_sample=0):
        sig=[]
        label=[]
        for i in range(n_sample):
            theta=random.randint(10, 100)
            x=Lr*k*np.cos(theta*np.pi/180)
            noise = np.random.randn(1,N) 	#产生N(0,1)噪声数据
            noise = noise-np.mean(noise) 								#均值为0
            signal_power = np.linalg.norm( x )**2 / x.size	#此处是信号的std**2
            noise_variance = signal_power/np.power(10,(SNR/10))         #此处是噪声的std**2
            noise = (np.sqrt(noise_variance) / np.std(noise) )*noise    ##此处是噪声的std**2
            signal_noise = noise + x
            x = signal_noise[0]
            # Pr1=np.cos(x)
            Pr1_i=np.sin(x)
            sig.append(Pr1_i)
            label.append(theta)
            
        data_train = np.array(sig[0:int(len(sig)*proportion)])
        label_train = np.array(label[0:int(len(label)*proportion)])
        data_val = np.array(sig[int(len(sig)*proportion):len(sig)])
        label_val = np.array(label[int(len(sig)*proportion):len(sig)])
        return (data_train, label_train), (data_val, label_val)
    
    # # Classification
    # num_nodes = 2  # Number of enhancement nodes.
    # regular_para = 1  # Regularization parameter.
    # weight_random_range = [-1, 1]  # Range of random weights.
    # bias_random_range = [0, 1]  # Range of random weights.
    # num_layer = 2  # Number of hidden layers

    # train, val, num_class = prepare_data_classify(0.8)
    # deep_rvfl = EnsembleDeepRVFL(n_nodes=num_nodes, lam=regular_para, w_random_vec_range=weight_random_range,
    #                      b_random_vec_range=bias_random_range, activation='relu', n_layer=num_layer, same_feature=False,
    #                      task_type='classification')
    # deep_rvfl.train(train[0], train[1], num_class)
    # prediction, proba = deep_rvfl.predict(val[0])
    # accuracy = deep_rvfl.eval(val[0], val[1])
    # print('Acc:', accuracy)


    # Regression
    f = 500
    c = 1500
    lmbda = c / f  #波长
    k=2*np.pi*f/c   #波束
    N=21
    Lr=np.linspace(0,10,N)  #阵列2米，每0.1米放置1个阵元
    
    num_nodes = 500  # Number of enhancement nodes. 10
    regular_para = 1  # Regularization parameter.
    weight_random_range = [-1, 1]  # Range of random weights.
    bias_random_range = [0, 1]  # Range of random weights.
    num_layer = 8  # Number of hidden layers

    time = datetime.datetime.now()
    train, val = data_generation(proportion=0.7,SNR=0,n_sample=5000)
    # train, val = prepare_data_regression(0.8)
    
    deep_rvfl = EnsembleDeepRVFL(n_nodes=num_nodes, lam=regular_para, w_random_vec_range=weight_random_range,
                         b_random_vec_range=bias_random_range, activation='relu', n_layer=num_layer, same_feature=False,
                         task_type='regression')
    deep_rvfl.train(train[0], train[1], 0)
    prediction = deep_rvfl.predict(val[0])
    mae = deep_rvfl.eval(val[0], val[1])
    print('MAE={}\nTime={}'.format(100-mae,datetime.datetime.now()-time))
