from sklearn import datasets
import numpy as np
np.random.seed(1)

from nets import MLP

def binary_test():
    data = datasets.make_blobs(n_samples=1000, centers=2)
    X = data[0]
    y = np.expand_dims(data[1], 1) # binary classification
    print('X: {}, y: {}'.format(X.shape, y.shape))

    net = MLP(2, 1, [4, 4],
             activation='sigmoid',
             final_activation='sigmoid',
             loss='mse')
    net.train(X, y, batch_size=16, epochs=25, lr=0.1)

def digits_test():
    data = datasets.load_digits()
    X, y = data['data'], data['target']
    num_classes = 10
    y = np.eye(num_classes)[y] # onehot encoded
    print('X: {}, y: {}'.format(X.shape, y.shape))

    net = MLP(64, num_classes, hidden_channels=[32, 32],
              activation='tanh',
              final_activation='sigmoid',
              loss='mse')
    print(net)
    net.train(X, y, batch_size=32, epochs=100, lr=0.001)

if __name__ == '__main__':
    # binary_test()
    digits_test()