from sklearn import datasets
import numpy as np
np.random.seed(1)

from nets import Net, MLP
from layers import Linear

def moons_test():
    print('-' * 10, 'moons test', '-' * 10)
    data = datasets.make_moons(n_samples=1000)
    X = data[0]
    y = np.expand_dims(data[1], 1) # binary classification
    print('X: {}, y: {}'.format(X.shape, y.shape))

    net = Net([Linear(2, 8, act='sigmoid'),
               Linear(8, 1, act='sigmoid')],
               loss='mse')
    print(net)

    net.train(X, y, batch_size=64, epochs=100, lr=0.5)
    print('-' * 35)

def digits_test():
    print('-' * 10, 'digits test', '-' * 10)
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

    net.train(X, y, batch_size=32, epochs=50, lr=0.01)
    print('-' * 35)

if __name__ == '__main__':
    moons_test()
    print()
    digits_test()