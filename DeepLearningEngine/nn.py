from DeepLearningEngine.backpropagation import Value
import numpy as np

class Neuron:
    def __init__(self, n_in):
        self.w = [Value(x) for x in np.random.rand(n_in)]
        self.b = Value(np.random.rand(1))

    def forward(self, input):
        assert(len(input) == len(self.w))
        net = np.sum([self.w[i] * input[i] for i in range(len(input))]) + self.b
        return net.leaky_relu()
        
    def params(self):
        return [self.b] + self.w

class Layer:
    def __init__(self, n_in, n_out):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]

    def forward(self, input):
        return [x.forward(input) for x in self.neurons]

    def params(self):
        params = []
        for neuron in self.neurons:
            params = params + neuron.params()

        return params

class MLP:
    def __init__(self, n_in, ns_out:list):
        nums = [n_in] + ns_out
        self.layers = [Layer(nums[i], nums[i+1]) for i in range(len(ns_out))]

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output[0] if len(output) == 1 else output

    def params(self):
        params = []
        for layer in self.layers:
            params = params + layer.params()

        return params