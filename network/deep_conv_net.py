from collections import OrderedDict
import pickle
import numpy as np
from network.layer import *

class DeepConvNet:
    """
    (class) DeepConvNet
    ---------------------
    - The deep convolutional neural network
    - Architecture is as follow
      - Conv -> ReLU -> Conv -> Relu -> Pooling\n
      -> Conv -> ReLU -> Conv -> Relu -> Pooling\n
      -> Conv -> ReLU -> Conv -> Relu -> Pooling\n
      -> Affine -> ReLU -> Dropout -> Affine -> Dropout -> Softmax

    Parameter
    ---------
    - input_dim : input dimension information on the dataset
    - conv_param_1 : 1st convolution layer parameters
    - conv_param_2 : 2nd convolution layer parameters
    - conv_param_3 : 3rd convolution layer parameters
    - conv_param_4 : 4th convolution layer parameters
    - conv_param_5 : 5th convolution layer parameters
    - conv_param_6 : 6th convolution layer parameters
    - num_hidden_node : number of hidden node
    - num_output_node : number of output node
    """
    # Object initializer
    def __init__(self, input_dim=(1, 28, 28), 
                conv_param_1={'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1},
                conv_param_2={'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1},
                conv_param_3={'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1},
                conv_param_4={'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1},
                conv_param_5={'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1},
                conv_param_6={'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1},
                num_hidden_node=50, num_output_node=10):
        # Initialize parameters
        pre_node_nums = np.array([1 * 3 * 3, 16 * 3 * 3, 16 * 3 * 3, 32 * 3 * 3, 32 * 3 * 3, 64 * 3 * 3, 64 * 3 * 3, num_hidden_node])
        weight_init_scales = np.sqrt(2.0 / pre_node_nums)  # ReLU를 사용할 때의 권장 초깃값
    
        # Initialize network parameters
        self.params = {}
        pre_channel_num = input_dim[0]
        for idx, conv_param in enumerate([conv_param_1, conv_param_2, conv_param_3, conv_param_4, conv_param_5, conv_param_6]):
            filter_num = conv_param['filter_num']
            filter_size = conv_param['filter_size']
            self.params['W' + str(idx + 1)] = weight_init_scales[idx] * np.random.randn(filter_num, pre_channel_num, filter_size, filter_size)
            self.params['b' + str(idx + 1)] = np.zeros(filter_num)
            pre_channel_num = filter_num
        self.params['W7'] = weight_init_scales[6] * np.random.randn(64 * 3 * 3, num_hidden_node)
        self.params['b7'] = np.zeros(num_hidden_node)
        self.params['W8'] = weight_init_scales[7] * np.random.randn(num_hidden_node, num_output_node)
        self.params['b8'] = np.zeros(num_output_node)

        # Create network architecture
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], conv_param_1['stride'], conv_param_1['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'], conv_param_2['stride'], conv_param_2['pad'])
        self.layers['Relu2'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers['Conv3'] = Convolution(self.params['W3'], self.params['b3'], conv_param_1['stride'], conv_param_1['pad'])
        self.layers['Relu3'] = Relu()
        self.layers['Conv4'] = Convolution(self.params['W4'], self.params['b4'], conv_param_2['stride'], conv_param_2['pad'])
        self.layers['Relu4'] = Relu()
        self.layers['Pool2'] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers['Conv5'] = Convolution(self.params['W5'], self.params['b5'], conv_param_1['stride'], conv_param_1['pad'])
        self.layers['Relu5'] = Relu()
        self.layers['Conv6'] = Convolution(self.params['W6'], self.params['b6'], conv_param_2['stride'], conv_param_2['pad'])
        self.layers['Relu6'] = Relu()
        self.layers['Pool3'] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers['Affine1'] = Affine(self.params['W7'], self.params['b7'])
        self.layers['Relu2'] = Relu()
        self.layers['Dropout1'] = Dropout(0.5)
        self.layers['Affine2'] = Affine(self.params['W8'], self.params['b8'])
        self.layers['Dropout2'] = Dropout(0.5)

        self.lastLayer = SoftmaxWithLoss()

    # Predict a response
    def predict(self, x, train_flag=False):
        for layer in self.layers.values():
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flag)
            else:
                x = layer.forward(x)

        return x

    # Calculate a loss value
    def loss(self, x, t):
        y = self.predict(x, train_flag=True)

        return self.lastLayer.forward(y, t)

    # Calculate an accuracy
    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        acc = 0.0
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx, train_flag=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)
        
        return acc / x.shape[0]

    # Calculate numerical gradients
    def numerical_gradient(self, x, t):
        # Do forward computations
        loss_W = lambda W: self.loss(x, t)

        # Calculate gradients
        grads = {}
        for idx in (1, 2, 3, 4, 5, 6, 7, 8):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])

        return grads

    # Calculate gradients using backpropagations
    def backprop_gradient(self, x, t):
        # Do forward computations
        self.loss(x, t)

        # Do backward computations
        dout = 1
        dout = self.lastLayer.backward(dout)        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # Save the gradients
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Conv2'].dW, self.layers['Conv2'].db

        grads['W3'], grads['b3'] = self.layers['Conv3'].dW, self.layers['Conv3'].db
        grads['W4'], grads['b4'] = self.layers['Conv4'].dW, self.layers['Conv4'].db

        grads['W5'], grads['b5'] = self.layers['Conv5'].dW, self.layers['Conv5'].db
        grads['W6'], grads['b6'] = self.layers['Conv6'].dW, self.layers['Conv6'].db

        grads['W7'], grads['b7'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W8'], grads['b8'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads

    def save_params(self, file_name='params.pkl'):
        # Save parameters
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name='params.pkl'):
        # Load parameters
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5', 'Conv6', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i + 1)]
            self.layers[key].b = self.params['b' + str(i + 1)]