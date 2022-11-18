import torch

# define model architecture
class Gabonisator(torch.nn.Module):
    # run super and define layers
    def __init__(self, n_features, n_outputs):
        super(Gabonisator, self).__init__()
        # define first hidden layer | input -> 64 nodes
        self.hidden1 = torch.nn.Linear(n_features, 64)
        # define second hidden layer | 64 nodes -> 64 nodes
        self.hidden2 = torch.nn.Linear(64, 64)
        # define output layer | 64 nodes -> output
        self.output = torch.nn.Linear(64, n_outputs)

    # define forward prop
    def forward(self, X):
        # first hidden layer + relu activation function
        y_pred = self.hidden1(X)
        y_pred = torch.relu(y_pred)
        # second hidden layer + relu activation function
        y_pred = self.hidden2(y_pred)
        y_pred = torch.relu(y_pred)
        # output layer + softmax to normalize results
        y_pred = self.output(y_pred)
        y_pred = torch.softmax(y_pred, dim=1)
        # return calculated y values
        return y_pred