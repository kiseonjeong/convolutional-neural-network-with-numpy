from activation_function import actFunc
from dataset.mnist import *

# The test neural network with three layers
class testNeuralNet:
    # Object initializer
    def __init__(self):
        self.network = {}
        self.func = actFunc()

    # Get the dataset
    def get_data(self):
        (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
        return x_test, t_test

    # Initialize a neural network
    def init_network(self):
        with open("sample_weight.pkl", 'rb') as f:
            self.network = pickle.load(f)

    # Do forward computations
    def forward(self, x):
        W1, W2, W3 = self.network['W1'], self.network['W2'], self.network['W3']
        b1, b2, b3 = self.network['b1'], self.network['b2'], self.network['b3']

        a1 = np.dot(x, W1) + b1
        z1 = self.func.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = self.func.sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        y = self.func.identity_function(a3)

        return y

    # Predict a response
    def predict(self, x):
        W1, W2, W3 = self.network['W1'], self.network['W2'], self.network['W3']
        b1, b2, b3 = self.network['b1'], self.network['b2'], self.network['b3']

        a1 = np.dot(x, W1) + b1
        z1 = self.func.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = self.func.sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        y = self.func.softmax(a3)

        return y

    
