import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import random as rand
import numpy as np
import math
import copy


def sigmoid_1d(a):

    return 1./ (1. + math.exp(-a))

class NeuralNet:

    """ Implements a neural network with flexible number of layers and units in each layer
        Create net by providing the number of nodes in each layer like this:
        [n_units_input, n_units_input, ... , n_units_output]
    """

    def __init__(self, nodes):

        self.net = [None] * len(nodes)
        for i in range(0, len(nodes)):
            if i <= (len(nodes)-2): self.net[i] = [0] * (nodes[i] + 1)
            else: self.net[i] = [0] * (nodes[i])


        weights_1 = np.full((3,2), -2.25)
        weights_2 = np.full((3,1), +1.35)
        self.weights = [weights_1, weights_2]

    def train(self, training_data, labels, alpha = 0.005):

        if len(training_data[0]) != nodes[0]:
            print("dimension of data points doesnt match the initialised net!!")
            return

        index = 0
        for datum in training_data:

            ## feed net with data and compute hidden and output layers

            for i in range(0, len(self.net[0])-1):                  # go over all input nodes x_i
                self.net[0][i] = datum[i]                           # initialise the input layer
            self.net[0][len(self.net[0])-1] = 1.0                   # initialise the input bias

            for j in range(0, len(self.net[1])-1):                  # go over all hidden layer nodes (currently assuming only one hidden layer)
                sum = 0.0
                for i in range(0, len(self.net[0])-1):              # go over all input layer nodes (currently assuming only one hidden layer)
                    sum += self.weights[0][i][j]*(self.net[0][i])   # calculate a_j = sum_i[x_i*weight_i,j]
                self.net[1][j] = math.tanh(sum)                     # calculate z_j = tanh(a_j)
            self.net[1][len(self.net[1])-1] = 1.0                   # z_bias = 1.0

            for k in range(0, len(self.net[2])):                    # go over all the output layer  nodes
                sum = 0.0
                for j in range(0, len(self.net[1])):                # go over all input layer nodes (currently assuming only one hidden layer)
                    sum += self.weights[1][j][k]*(self.net[1][j])   # calculate a_k = sum_j[z_j*w_j,k]
                self.net[2][k] = sigmoid_1d(sum)                    # calculate y_k = sigmoid(a_k)


            ## calculation of the new weights

            grad_E = copy.deepcopy(self.weights)                    # initalize grad_E as weights to get the right dimensions
            delta = copy.deepcopy(self.net)                         # initalize delta as weights to get the right dimensions + imput layer

            ## calculation of the delta terms
            ## delta[layer][node] | the first layer is not needed!
            ## delta_k
            for k in range(0, len(self.net[2])):                    # go over all the output layer  nodes
                delta[2][k] = (self.net[2][k] - labels[index])      # calculate y_k = sigmoid(a_k)

            ## delta_j
            for j in range(0, len(self.net[1])):                    # go over all hidden layer nodes (currently assuming only one hidden layer)
                sum = 0.0
                for k in range(0, len(self.net[2])):                # go over all input layer nodes (currently assuming only one hidden layer)
                    sum += self.weights[0][j][k]*(delta[2][k])      # calculate a_j = sum_i[x_i*weight_i,j]
                delta[1][j] = (1.0 - (self.net[1][j]**2.0)) * sum   # calculate z_j = tanh(a_j)

            ## calculation of the gradient of the error function
            ## grad_E[layer][outgoing node][ingoing node]
            ## grad_E[0]
            for j in range(0, len(self.net[1])-1):                  # go over all possible delta_j
                for i in range(0, len(self.net[0])):                # go over all nodes x_i
                    grad_E[0][i][j] = delta[1][j]*self.net[0][i]    # calculate grad E_ji = delta_j * x_i

            ## grad_E[1]
            for k in range(0, len(self.net[2])):                    # go over all possible delta_k
                for j in range(0, len(self.net[1])):                # go over all hidden nodes z_j
                    grad_E[1][j][k] = delta[2][k]*self.net[0][j]    # calculate grad E_kj = delta_k * z_j

            for layer in range(0, len(self.weights)):
                for neuron in range(0, len(self.weights[layer])):
                    for weight in range(0, len(self.weights[layer][neuron])):
                        self.weights[layer][neuron][weight] = self.weights[layer][neuron][weight] - alpha*grad_E[layer][neuron][weight]

            index += 1

    def predict(self, test_data, test_labels):

        prediction = []

        index = 0
        pred = 0
        for datum in test_data:

            ## feed net with data and compute hidden and output layers

            for i in range(0, len(self.net[0])-1):                  # go over all input nodes x_i
                self.net[0][i] = datum[i]                           # initialise the input layer
            self.net[0][len(self.net[0])-1] = 1.0                   # initialise the input bias

            for j in range(0, len(self.net[1])-1):                  # go over all hidden layer nodes (currently assuming only one hidden layer)
                self.net[1][j] = 0.
                sum = 0.0
                for i in range(0, len(self.net[0])-1):              # go over all input layer nodes (currently assuming only one hidden layer)
                    sum += self.weights[0][i][j]*(self.net[0][i])   # calculate a_j = sum_i[x_i*weight_i,j]
                self.net[1][j] = math.tanh(sum)                     # calculate z_j = tanh(a_j)
            self.net[1][len(self.net[1])-1] = 1.0                   # z_bias = 1.0

            for k in range(0, len(self.net[2])):                    # go over all the output layer  nodes
                sum = 0.0
                for j in range(0, len(self.net[1])):              # go over all input layer nodes (currently assuming only one hidden layer)
                    sum += self.weights[1][j][k]*(self.net[1][j])   # calculate a_k = sum_j[z_j*w_j,k]
                self.net[2][k] = sigmoid_1d(sum)                    # calculate y_k = sigmoid(a_k)


            if self.net[2][k] >= 0.5:
                classification = 1
            else:
                classification = 0
            performance = test_labels[index] - classification
            prediction.append((classification, performance))

            pred += (1-abs(performance))

            index += 1

        pred = pred / len(test_data)
        pred = pred * 100.

        print("Prediction chance = ", pred, "%")

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

    netti.train(data, t)


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
