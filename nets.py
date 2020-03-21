import numpy as np

from layers import Linear
from functions import get_activation, get_loss, accuracy

class Net:
    def __init__(self, layers, loss='mse'):
        self.layers = layers
        self.criterion = get_loss(loss)

    @property
    def num_layers(self):
        return len(self.layers)

    def __str__(self):
        s = '{}(\n'.format(self.__class__.__name__)
        for i, l in enumerate(self.layers):
            s += '  ({}): {},\n'.format(i, str(l))
        s += ')'
        return s

    def train(self, X, y, epochs=0, batch_size=1, lr=0.1, verbose=True):
        num_samples = len(X)
        for e in range(epochs):
            idx = np.random.permutation(num_samples)
            X_, y_ = X[idx], y[idx]

            metrics = dict(accuracies=[], losses=[])
            for i in range(0, len(X), batch_size): # NOTE: last batch size not equalized
                X_batch = X_[i:i + batch_size].T
                y_batch = y_[i:i + batch_size].T
                # input X_batch has shape (num_features, num_samples), with the batched index last
                # target y_batch has shape (num_classes, num_samples), one-hot encoding

                y_pred, cache = self.forward(X_batch)
                loss = self.criterion(y_pred, y_batch)
                self.update(y_pred, y_batch, cache, num_samples, lr)

                acc = accuracy(y_pred, y_batch)
                metrics['accuracies'].append(acc)
                metrics['losses'].append(loss)

            if verbose: 
                print('[epoch {}] acc: {:.3f}, loss: {:.3f}'.format(e,
                                                                    np.mean(metrics['accuracies']),
                                                                    np.mean(metrics['losses'])
                                                                   ))

    def forward(self, x):
        cache = dict(z_s=[], a_s=[x]) # NOTE: storing initial input
        a_cur = x
        for i, layer in enumerate(self.layers):
            z, a = layer.forward(a_cur)
            cache['z_s'].append(z) # indexes current layer output
            cache['a_s'].append(a) # indexes prev layer activation output
            a_cur = a
        return a_cur, cache

    def update(self, y_pred, y, cache, num_samples, lr):
        da = self.criterion(y_pred, y, deriv=True)
        for l in reversed(range(self.num_layers)): # backprop
            a_prev = cache['a_s'][l]
            z = cache['z_s'][l]
            da, dw, db = self.layers[l].backward(a_prev, z, da)

            # gradient descent step
            self.layers[l].weight -= lr * dw
            self.layers[l].bias -= lr * db

class MLP(Net):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=[],
                 activation='sigmoid',
                 final_activation='sigmoid',
                 **kwargs,
                 ):
        self.channels = [in_channels] + hidden_channels + [out_channels]
        self.activation = activation
        self.final_activation = final_activation
        layers = self.make_layers()
        super().__init__(layers, **kwargs)

    def make_layers(self):
        layers = []
        for i in range(1, len(self.channels)):
            x = self.channels[i - 1]
            y = self.channels[i]
            act = self.final_activation if i == (len(self.channels) - 1) else self.activation
            layers.append(Linear(x, y, act=act))
        return layers

# class CNN(Net):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#     def make_layers(self):
#         pass