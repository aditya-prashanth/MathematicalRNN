import torch

class linear:

    # represents a single fully connected layer in the network

    # l1 is 'x' (the input layer), l2 is 'a' (the output layer)
    # l2 is the output layer from the 'a = Wx + b' function
    # l2_adj is l2 after the reLU function has been applied

    # W is the weight matrix where the rows represent the connections to the 'a' layer
    # and the columns represent the connections to the 'x' layer, 
    # i.e W[1,5] is the connection between the 5th neuron in the 'x' layer to the 1st neuron in 'a'

    def __init__(self, l1_size, l2_size):
        self.l1 = torch.zeros(l1_size, dtype=torch.float)
        self.l2 = torch.zeros(l2_size, dtype=torch.float)
        self.l2_adj = torch.zeros(l2_size, dtype=torch.float)

        # W initialized to a normal distribution with a variance of 2 / the size of x
        # b initialized to 0.01 to prevent dead neurons
        self.W = torch.normal(0, torch.sqrt(torch.tensor(2.0 / l1_size)), (l2_size, l1_size))
        self.b = torch.zeros(l2_size, dtype=torch.float) + 0.01
    
class RNN:

    # RNN has 4 layers, the input layer, two hidden layers, and the output layer

    def __init__(self, input_size, output_size):
        self.fc1 = linear(input_size, 32)
        self.fc2 = linear(32, 32)
        self.fc3 = linear(32, output_size)

    #represents 1 forward pass through the network

    def forward_pass(self, x):

        # copies the input to the first input layer of the network
        # then performs the 'a = Wx + b' operation, and then performs reLU to the output
        self.fc1.l1.copy_(x)
        self.fc1.l2 = torch.matmul(self.fc1.W, self.fc1.l1) + self.fc1.b
        self.fc1.l2_adj = torch.max(torch.zeros_like(self.fc1.l2), self.fc1.l2)

        # copies the 'a' from the last fc layer to the 'x' of the 2nd layer in-place
        # the relu function simply eliminates negative values by replacing them with zero
        self.fc2.l1.copy_(self.fc1.l2_adj)
        self.fc2.l2 = torch.matmul(self.fc2.W, self.fc2.l1) + self.fc2.b
        self.fc2.l2_adj = torch.max(torch.zeros_like(self.fc2.l2), self.fc2.l2)

        self.fc3.l1.copy_(self.fc2.l2_adj)
        self.fc3.l2 = torch.matmul(self.fc3.W, self.fc3.l1) + self.fc3.b
        self.fc3.l2_adj = torch.max(torch.zeros_like(self.fc3.l2), self.fc3.l2)

    # helper function for the derivative of reLU: max(0, x), is 0 if negative and 1 if positive
    def relu_derivative(self, x):
        return (x > 0).float()
    
    # performs one backward pass through the network, readjusting values of W and b

    # values are adjusted base on the derivative of the loss function with respect to the value
    # A number of derivatives are used to calculate the learning rates which I'll discuss below

    # W = W - alpha * dL/dW -- alpha : learning rate , dL/dW : derivative of loss with respect to weight
    # dL/dW = dL/dz * (dz/dx => d/dx z => d/dx Wx + b => d/dx x^t -- dL/dz : derivative of loss with respect to a before the reLu function , x^t : the transpose of x

    # b = b - alpha * dL/db -- dL/db : d of loss with respect to bias
    # dL/db = dL/dz * (dz/db => d/db (a - y)^2 + b => 1) => dL/dz

    # dL/dz = dL/da * da/dz -- dL/da : d of loss with respect to a, da/dz: d of a with respect to a before the relu
    # dL/da = d/da (a - y)^2 + b  => 2*(a - y) -- a : the output vector y : the test vector
    # da/dz = d/dz relu(z) => relu'(z) -- z : a before relu is applied

    def backward_pass(self, learning_rate, y):
        dLdZ = self.dLdZ(self.fc3, y)
        self.fc3.W -= learning_rate * torch.matmul(dLdZ.unsqueeze(1), self.fc3.l1.unsqueeze(0))
        self.fc3.b -= learning_rate * dLdZ

        dLdX = torch.matmul(self.fc3.W.T, dLdZ)
        dLdZ = dLdX * self.relu_derivative(self.fc2.l2)
        self.fc2.W -= learning_rate * torch.matmul(dLdZ.unsqueeze(1), self.fc2.l1.unsqueeze(0))
        self.fc2.b -= learning_rate * dLdZ

        dLdX = torch.matmul(self.fc2.W.T, dLdZ)
        dLdZ = dLdX * self.relu_derivative(self.fc1.l2)
        self.fc1.W -= learning_rate * torch.matmul(dLdZ.unsqueeze(1), self.fc1.l1.unsqueeze(0))
        self.fc1.b -= learning_rate * dLdZ

    def dLdZ(self, layer, cost):
        dLdA = 2 * (layer.l2_adj - cost)
        dAdZ = self.relu_derivative(layer.l2)
        return dLdA * dAdZ
