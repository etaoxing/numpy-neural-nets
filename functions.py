import numpy as np

def linear(z, deriv=False):
    if deriv:
        return np.ones(z.shape)
    return z

def sigmoid(z, deriv=False):
    if deriv:
        return sigmoid(z) * (1 - sigmoid(z))
    return 1. / (1. + np.exp(-z))

def tanh(z, deriv=False):
    if deriv:
        return 1. - np.tanh(z) ** 2
    return np.tanh(z)

def relu(z, deriv=False):
    if deriv:
        return 1. * (z > 0)
    return np.maximum(0, z)

def softmax(z, deriv=False):
    if deriv:
        # J_(i,j) = p_i (delta_(i,j) - p_j)
        raise NotImplementedError
    e_z = np.exp(z - np.max(z)) # shift by max for numerical stability, https://cs231n.github.io/linear-classify/#softmax
    return e_z / np.sum(e_z, axis=1, keepdims=True)

def log_softmax(z, deriv=False):
    if deriv:
        raise NotImplementedError
    shifted = z - np.max(z)
    return shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))

def get_activation(f):
    if f == 'linear':
        return linear
    elif f == 'sigmoid':
        return sigmoid
    elif f == 'tanh':
        return tanh
    elif f == 'relu':
        return relu
    elif f == 'softmax':
        return softmax
    else:
        raise NotImplementedError

def mse(y_pred, y_true, deriv=False):
    if deriv:
        return y_pred - y_true
    return np.mean((y_pred - y_true) ** 2)

def cross_entropy(y_pred, y_true, deriv=False): # NOTE: use with logits (final_activation='linear')
    n = y_pred.shape[1]
    if deriv:
        y = y_true.argmax(axis=0) # NOTE: get class labels, convert from onehot encoding
        delta = softmax(y_pred)
        delta[y, np.arange(n)] -= 1
        return delta
    y_pred = log_softmax(y_pred)
    cost = -1. * np.sum(y_pred * y_true) / n
    return cost

def get_loss(f):
    if f == 'mse':
        return mse
    elif f == 'cross_entropy':
        raise NotImplementedError
        return cross_entropy
    else:
        raise NotImplementedError

def accuracy(y_pred, y_true):
    if len(y_pred) == 1: # binary classification
        y_true = y_true[0]
        y_pred = (y_pred > 0.5).astype(np.int)[0]
    else:
        y_true = np.argmax(y_true, axis=0) # NOTE: one-hot encoded
        y_pred = np.argmax(y_pred, axis=0)

    acc = np.sum(y_pred == y_true) / len(y_true)
    return acc