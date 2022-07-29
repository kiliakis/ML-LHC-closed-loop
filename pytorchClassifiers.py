import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from functools import partial
from tensorflow import keras


def get_keras_nn(n_inputs, n_outputs, name='phase_regressor', activation='relu',
                 layers=[140, 110], kernel_size=5, stride=5, trim_edges=120,
                 dropout=0.0, with_norm=False):
    # initialize model
    model = keras.models.Sequential(name=name)
    # just set input shape
    model.add(keras.Input(shape=n_inputs))
    # crop the edges
    model.add(keras.layers.Cropping1D(cropping=trim_edges, name='Crop'))
    # smoothing layer
    model.add(keras.layers.AveragePooling1D(pool_size=kernel_size, strides=stride,
                                            name='AvgPool'))
    # normalize input
    model.add(keras.layers.BatchNormalization(axis=2, name='Norm_0'))
    # flatten before the dense layers
    model.add(keras.layers.Flatten(name='Flat'))
    # for every layer
    for i, layer_size in enumerate(layers):
        # Add dense layer with activation_func
        model.add(keras.layers.Dense(layer_size, activation=activation,
                                     name=f'Dense_{i+1}'))
        # Add batch normalization
        if with_norm:
            model.add(keras.layers.LayerNormalization(
                axis=1, name=f'Norm_{i+1}'))
        # Add dropout
        if dropout > 0 and dropout < 1:
            model.add(keras.layers.Dropout(dropout, name=f'Dropout_{i+1}'))
    model.add(keras.layers.Dense(n_outputs, name=f'Output'))
    return model


def plot_history(history, lines=('loss', 'val_loss'), epoch=(-1, -1)):
    fig = plt.figure()

    if epoch[0] == -1:
        start_idx = 0
    else:
        start_idx = epoch[0]

    for line in lines:
        assert line in history
        y = history[line]

        if epoch[1] == -1:
            end_idx = len(y)
        else:
            end_idx = epoch[1]
        y = y[start_idx: end_idx]

        plt.plot(np.arange(start_idx, end_idx), y, label=line)

    plt.xlabel('epoch')
    plt.legend()
    plt.show()
    plt.close()

    return fig


def standardize(x):
    mean = x.mean(1, keepdim=True)
    std = x.std(1, unbiased=False, keepdim=True)
    x = (x - mean) / std
    return x


class NetBase(nn.Module):
    def __init__(self, n_inputs, n_outputs, *args, **kwargs):
        # This needs to be implemented
        super(NetBase, self).__init__()
        # absolutely required attributes
        self.history = {'train_loss': [],
                        'val_loss': [],
                        'train_accuracy': [],
                        'val_accuracy': []}
        self.name = ''
        self.save_best = False
        self.verbose = 0
        self.layers = []

        return

    def forward(self, x):
        # This needs to be implemented too
        # all the functions to be applied should be in a list
        for layer in self.layers:
            x = layer(x)
        return x

    def __get_accuracy(self, log_probs, labels):
        ps = torch.exp(log_probs)
        top_p, top_class = ps.topk(1, dim=1)
        n_correct = top_class == labels.view(*top_class.shape)
        acc = (n_correct.sum().item() * 100.0 / len(labels))
        return acc

    def evaluate(self, X, labels):
        top_class = self.get_predictions(X, labels)
        n_correct = top_class == labels.view(*top_class.shape)
        acc = (n_correct.sum().item() * 100.0 / len(labels))
        return acc

    def get_predictions(self, X, encoder=None):
        self.eval()
        with torch.no_grad():
            yhat = self(X)
        _, yhat = torch.exp(yhat).topk(1, dim=1)
        yhat = yhat.ravel().numpy()
        if encoder:
            yhat = encoder.inverse_transform(yhat)
        return yhat

    def save(self, optimizer, filename=None):
        if not filename:
            filename = self.name
        torch.save({'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, filename)

    def load(self, optimizer, filename):
        checkpoint = torch.load(filename)
        if 'optimizer_state_dict' not in checkpoint:
            self.load_state_dict(checkpoint)
        else:
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def train_net(self, optimizer, criterion, X_train, y_train, X_test, y_test, num_epochs=100):
        l_train_loss = []
        l_train_accuracy = []
        l_test_loss = []
        l_test_accuracy = []
        min_test_loss = np.Inf
        best_model = self
        for epoch in range(num_epochs):
            # put the model in training mode
            self.train()

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # forward pass: compute predicted outputs by passing inputs to the model
            output_train = self(X_train)
            l_train_accuracy.append(self.__get_accuracy(output_train, y_train))

            # calculate the loss
            train_loss = criterion(output_train, y_train)
            l_train_loss.append(train_loss.item())

            # backward pass: compute gradient of the loss with respect to model parameters
            train_loss.backward()

            # perform a single optimization step (parameter update)
            optimizer.step()

            with torch.no_grad():
                output_test = self(X_test)
                test_loss = criterion(output_test, y_test)
                l_test_loss.append(test_loss.item())
                l_test_accuracy.append(
                    self.__get_accuracy(output_test, y_test))

            if l_test_loss[-1] < min_test_loss:
                if self.verbose > 0:
                    print(
                        f"Epoch {epoch + 1}/{num_epochs}: Test loss decreased ({min_test_loss:.6f} --> {l_test_loss[-1]:.6f})")
                best_model = self
                min_test_loss = l_test_loss[-1]

            if (epoch + 1) % 50 == 0 and self.verbose > 0:
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {l_train_loss[-1]:.4f}, Test Loss: {l_test_loss[-1]:.4f}")

        if self.save_best:
            best_model.save(optimizer, f'models/{self.name}.pt')

        self.history['train_loss'] += l_train_loss
        self.history['train_accuracy'] += l_train_accuracy
        self.history['val_loss'] += l_test_loss
        self.history['val_accuracy'] += l_test_accuracy

        return l_train_loss, l_train_accuracy, l_test_loss, l_test_accuracy

    def plot_history(self, lines=('train_loss', 'val_loss'), epoch=(-1, -1)):
        fig = plt.figure()

        if epoch[0] == -1:
            start_idx = 0
        else:
            start_idx = epoch[0]

        for line in lines:
            assert line in self.history
            y = self.history[line]

            if epoch[1] == -1:
                end_idx = len(y)
            else:
                end_idx = epoch[1]
            y = y[start_idx: end_idx]

            plt.plot(np.arange(start_idx, end_idx), y, label=line)

        plt.xlabel('epoch')
        plt.legend()
        plt.show()
        plt.close()

        return fig


class Net(NetBase):
    def __init__(self, n_inputs, n_outputs, name='classifier', activation_func=F.gelu,
                 hidden_layers=(140, 110), avg_pool=(4, 4, 0), trim_edges=120, dropout=0.2,
                 with_batch_norm=False, save_best=True, scale_input=False, verbose=1):
        super(NetBase, self).__init__()

        self.history = {'train_loss': [],
                        'val_loss': [],
                        'train_accuracy': [],
                        'val_accuracy': []}
        self.name = name
        self.save_best = save_best
        self.verbose = verbose
        self.layers = []

        # Standardize the input samples across axis 1
        # This is preferred so that the network can generalize in new data.
        if scale_input:
            self.scaler = standardize
            self.layers.append(self.scaler)

        # A layer that trims out the edges
        self.center = nn.ConstantPad1d(-trim_edges, 0)
        self.layers.append(self.center)

        # An avgpooling layer to smoothen the curve
        n_inputs -= 2 * trim_edges
        if len(avg_pool) == 3:
            kernel_size, stride, padding = avg_pool
        else:
            kernel_size, stride = avg_pool
            padding = 0
        n_pool_out = (n_inputs + 2 * padding - kernel_size) // stride + 1
        self.avg_pool = nn.AvgPool1d(
            kernel_size=kernel_size, stride=stride, padding=padding)
        self.layers.append(self.avg_pool)

        # this loop creates all the dense hidden layers
        out_nodes = n_pool_out
        for i, hidden_nodes in enumerate(hidden_layers):
            in_nodes = out_nodes
            out_nodes = hidden_nodes
            setattr(self, f'fc{i}', nn.Linear(in_nodes, out_nodes))
            setattr(self, f'activate{i}', activation_func)

            self.layers += [getattr(self, f'fc{i}'),
                            getattr(self, f'activate{i}')]
            if with_batch_norm:
                setattr(self, f'batch_norm{i}', nn.BatchNorm1d(out_nodes))
                self.layers.append(getattr(self, f'batch_norm{i}'))
            setattr(self, f'dropout{i}', nn.Dropout(dropout))
            self.layers.append(getattr(self, f'dropout{i}'))

        # Output layer
        self.out_layer = nn.Linear(out_nodes, n_outputs)
        self.layers.append(self.out_layer)

        # final activation function
        self.activate_out = partial(F.log_softmax, dim=1)
        self.layers.append(self.activate_out)
