import random
from engine import Value

class Neuron:
    # Neuron is a unit defined by the number of inputs
    def __init__(self, n_inputs):
        self.weights = [Value(random.random()) for _ in range(n_inputs)]
        self.bias = Value(random.random())
    
    # Forward Pass on Neuron
    def __call__(self, x):
        assert(len(x) == len(self.weights))

        # compute wx + b
        result = 0.0
        for xi, wi in zip(self.weights, x):
            result += xi * wi
        result += self.bias

        # apply activation function
        return result.tanh()
    
    def parameters(self):
        return self.weights + [self.bias]

class Layer:
    # Defines two layers of neurons
    def __init__(self, n_inputs, n_outputs):
        self.layer = [Neuron(n_inputs) for _ in range(n_outputs)]
    
    # Given the inputs from the previous layer, compute outputs of the current layer
    def __call__(self, x):
        result = [neuron(x) for neuron in self.layer]
        return result[0] if len(result) == 1 else result

    def parameters(self):
        params = []
        for neuron in self.layer:
            params.extend(neuron.parameters())
        return params

class MLP:
    # Create an MLP with dims of all layers
    # dims: dimensions of all layers
    def __init__(self, dims):
        self.layers = [Layer(dims[x], dims[x+1]) for x in range(len(dims) - 1)]
    
    # forward pass through all layers of the network
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    # MSE loss
    def loss(self, ypred, ygt):
        sum = 0.0
        for yp, yg in zip(ypred, ygt):
            sum += (yp - yg) ** 2
        sum /= len(ypred)
        return sum

    # Return all parameters
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
    # Zero_out parameters
    def zero_out(self):        
        for p in self.parameters():
            p.grad = 0.0


# Testing it.
dims = [2, 4, 3, 1]
inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
expected = [0.0, 1.0, 1.0, 0.0]
network = MLP(dims)
preds = [network.forward(input) for input in inputs]

learning_rate = 0.05
for _ in range(5000):
    # Forward Pass
    preds = [network.forward(input) for input in inputs]
    loss = network.loss(preds, expected)

    # Zero-out + Backward Pass
    network.zero_out()
    loss.backward()

    # Update Parameters
    for p in network.parameters():
        p.data += -1 * learning_rate * p.grad

    if _ % 100 == 0:
        print(_, network.loss(preds, expected))