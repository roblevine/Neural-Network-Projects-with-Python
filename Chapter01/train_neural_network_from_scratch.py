import numpy as np

# A sigmoid function is any mathematical function whose graph has a characteristic S-shaped or sigmoid curve.
# A common example of a sigmoid function is the logistic function, which is defined in this function
# this tends to 0 as x tends to minus infinity, and tends to 1 as x tends to plus infinity. At zero it is 0.5.
def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        print("\nself.input:\n{}".format(self.input))


        self.weights1   = np.random.rand(self.input.shape[1],4) 
        print("\nself.weights1:\n{}".format(self.weights1))

        self.weights2   = np.random.rand(4,1)                 
        print("\nself.weights2:\n{}".format(self.weights2))

        self.y          = y
        print("\nself.y:\n{}".format(self.y))

        self.output     = np.zeros(self.y.shape)
        print("\nself.output:\n{}".format(self.output))
        

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2


if __name__ == "__main__":
    X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
    y = np.array([[0],[1],[1],[0]])
    nn = NeuralNetwork(X,y)

    for i in range(1500):
        nn.feedforward()
        nn.backprop()

    print("\nOutput:\n{}".format(nn.output))

