import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import random as rand
import numpy as np

def sigmoid(a):

    a = np.array(a)
    return 1./ (1 + np.exp(-a))

def tanh(a):

    a = np.array(a)
    return np.append(np.tanh(a), 1)

class NeuralNet:

    """ Implements a neural network with flexible number of layers and units in each layer
        Create net by providing the number of nodes in each layer like this:
        [n_units_input, n_units_input, ... , n_units_output]
    """

    def __init__(self, nodes,  activation_functions = [tanh, sigmoid]):

        self.net = []
        self.input_nodes = nodes[0]+1

        self.funcs = activation_functions

        for n_units in nodes:
            self.net.append([n_units+1]) # add bias to each layer
        #self.net[-1][0] -= 1            # except output layer

        self.weights = []                # [edge][m][d] || z_M = x * w_m = sum{x_d * w_md}

        for it in range(0, len(self.net) - 1):
            n_weights_in  = self.net[it][0]
            n_weights_out = self.net[it+1][0]-1
            vector = []
            for itt in range(0, n_weights_out):
                vector.append(np.random.random(n_weights_in))
            self.weights.append(vector)


    def train(self, training_data, labels, alpha = .2):

        if len(training_data[0]) != self.input_nodes - 1:
            print("dimension of data points doesnt match the initialised net!!")
            return

        index = 0

        for datum in training_data:

            ## feed net with data and compute hidden and output layers

            self.net[0] = np.append(datum, 1)
            it = 1

            for weight_matrix in self.weights:

                self.net[it] = []

                for vector in weight_matrix:
                    self.net[it].append(np.dot(vector, self.net[it-1]))

                self.net[it] = self.funcs[it-1](self.net[it])
                it += 1

            #print(self.net)

        ## -- compute gradients through back propagation --

            grad_E = []

            # output layer:

            delta = [[]]

            for node in self.net[-1]:

                delta[0].append(node - labels[index])

            ## hidden layers:

            for it in range(len(self.net)-2, 0, -1):

                delta_hidden = []
                j = 0 ## index of z

                for z in self.net[it]:

                    error_sum = 0
                    k = 0 ## index of error
                    for error in delta[it-1]:
                        error_sum += self.weights[it][k][j]
                        k += 1
                    delta_hidden.append((1 - z**2)*error_sum)

                delta.insert(0, delta_hidden[:-1])
                j += 1

            ## compute gradient

            for layer in range(0, len(self.net)-1):
                grad_layer = []
                for k in range(0, len(delta[layer])):
                    grad_neuron = []
                    for j in range(0, len(self.net[layer])):
                        grad_neuron.append(delta[layer][k]*self.net[layer][j])
                    grad_layer.append(np.array(grad_neuron))
                grad_E.append(grad_layer)

            #print(delta)
            print(grad_E)
            #print(self.weights)

        ## -- update weights --

            for layer in range(0, len(self.weights)):
                for neuron in range(0, len(self.weights[layer])):
                    self.weights[layer][neuron] = self.weights[layer][neuron] - alpha*grad_E[layer][neuron]

            print(self.weights)

            print("--")

        index += 1

    def predict(self, test_data, test_labels):

        prediction = []

        index = 0
        for datum in test_data:

            ## feed net with data and compute hidden and output layers

            self.net[0] = np.append(datum, 1)
            it = 1

            for weight_matrix in self.weights:

                self.net[it] = []

                for vector in weight_matrix:
                    self.net[it].append(np.dot(vector, self.net[it-1]))

                self.net[it] = self.funcs[it-1](self.net[it])
                it += 1

            classification =  1 if self.net[-1][0] >= 0.5 else 0
            performance = test_labels[index] - classification
            prediction.append((classification, performance))

            index += 1

        return prediction

    def print(self):

        for line in self.net:
            print(line)

        for matrix in self.weights:
            print(matrix)



def two_layered_neural_net():

    np.random.seed()
    n_test = 20

    # -- Generate Data --
    ## _1 and _2 denote the classes from which the points were drawn

    set_size = 10

    mu_1 = [2, 2]
    mu_2 = [1, -1]

    sigma_1 = [[0.8, 0.4], [0.4, 0.8]]
    sigma_2 = [[1.3, -0.7], [-0.7, 1.3]]

    t = np.append(np.full(set_size, 0), np.full(set_size, 1))  # 0 -> class 1, 1 -> class 2
    np.random.shuffle(t)

    #data = np.array([])

    for it in range(0, t.size):
        if t[it] == 0:
            x = np.random.multivariate_normal(mu_1, sigma_1, size=1)
        else:
            x = np.random.multivariate_normal(mu_2, sigma_2, size=1)

        if it == 0:
            data = x
        else:
            data = np.append(data, x, axis=0)

    ## -- train net --

    nodes = [2, 2, 1]
    netti = NeuralNet(nodes)
    #netti.print()

    netti.train(data, t)
    #netti.print()

    ## -- test net --

    rand.seed(0)
    n1_size = rand.randint(0, n_test)
    n2_size = n_test - n1_size

    x_1 = np.random.multivariate_normal(mu_1, sigma_1, size=n1_size)
    x_2 = np.random.multivariate_normal(mu_2, sigma_2, size=n2_size)

    test_data = np.append(x_1, x_2, axis=0)
    test_label = np.append(np.full(n1_size, 0), np.full(n2_size, 1))  # 0 -> class 1, 1 -> class 2

    print(netti.predict(test_data, test_label))

    ## -- plot --

def main():
    two_layered_neural_net()

if __name__ == "__main__":
    main()
