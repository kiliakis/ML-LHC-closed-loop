import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from matplotlib import pyplot as plt


class Net(nn.Module):
    def __init__(self, n_inputs, n_outputs, name='classifer', activation_func=F.gelu,
                 hidden_1=140, hidden_2=110, hidden_3=0,
                 avg_pool=(4, 4, 0), trim_edges=120, dropout=0.2,
                 with_batch_norm=False, save_best=True, verbose=1):
        super(Net, self).__init__()

        self.history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}
        self.activate = activation_func
        self.name = name
        self.save_best = save_best
        self.with_batch_norm = with_batch_norm
        self.verbose=verbose
        # A layer that trims out the edges
        self.center = nn.ConstantPad1d(-trim_edges, 0)
        n_inputs -= 2 * trim_edges

        # An avgpooling layer to smoothen the curve
        if len(avg_pool) == 3:
            kernel_size, stride, padding = avg_pool
        else:
            kernel_size, stride = avg_pool
            padding = 0
        n_pool_out = (n_inputs + 2 * padding - kernel_size) // stride + 1
        self.pool1 = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)

        # linear layer (n_pool_out -> hidden_1)
        self.fc1 = nn.Linear(n_pool_out, hidden_1)
        if with_batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(hidden_1)

        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        if with_batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(hidden_2)

        # linear layer (n_hidden -> n_outputs)
        if hidden_3 > 0:
            self.fc3 = nn.Linear(hidden_2, hidden_3)
            if with_batch_norm:
                self.batch_norm3 = nn.BatchNorm1d(hidden_3)

            self.fc4 = nn.Linear(hidden_3, n_outputs)
        else:
            self.fc4 = None
            self.fc3 = nn.Linear(hidden_2, n_outputs)

        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # remove some points from the edges
        x = self.center(x)
        # This performs the average pooling layer
        x = self.pool1(x)
        # add hidden layer, with activation function and dropout
        x = self.activate(self.fc1(x))
        if self.with_batch_norm:
            x = self.batch_norm1(x)
        x = self.dropout(x)

        # add hidden layer, with activation function and dropout
        x = self.activate(self.fc2(x))
        if self.with_batch_norm:
            x = self.batch_norm2(x)
        x = self.dropout(x)

        if self.fc4:
            x = self.activate(self.fc3(x))
            if self.with_batch_norm:
                x = self.batch_norm3(x)
            x = self.dropout(x)
            x = self.fc4(x)
        else:
            x = self.fc3(x)

        # Output layer with Softmax activation
        x = F.log_softmax(x, dim=1)
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

    def save(self, name='model.pt'):
        torch.save(self.state_dict(), name)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))

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
                l_test_accuracy.append(self.__get_accuracy(output_test, y_test))

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
            best_model.save(f'models/{self.name}.pt')

        self.history['train_loss'] += l_train_loss
        self.history['train_accuracy'] += l_train_accuracy
        self.history['val_loss'] += l_test_loss
        self.history['val_accuracy'] += l_test_accuracy

        return l_train_loss, l_train_accuracy, l_test_loss, l_test_accuracy

    def plot_history(self, lines=['train_loss', 'val_loss'], epoch=(-1, -1)):
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
