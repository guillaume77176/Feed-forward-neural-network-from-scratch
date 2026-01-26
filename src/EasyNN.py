###################################################################
# Basic implementation of Feed Forward Neural Network, from scratch #
###################################################################

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import pickle

"""
Activations functions
"""


def relu(z):
    return np.maximum(z, 0)


def relu_prime(z):
    return (z > 0).astype(int)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(z):
    z = z - np.max(z, axis=0, keepdims=True)        # avoid saturation of softmax
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


"""
Loss functions

(classification only)
"""


def binary_cross_entropy(y_true, y_pred):
    """
    y_true : 0 or 1
    y_pred : output scalar from sigmoid or others
    """
    return -(y_true*np.log(y_pred + 1e-12) + (1-y_true)*np.log(1-y_pred + 1e-12))


def cross_entropy(y_true: np.array, y_pred: np.array):
    """
    y_true : one-hot vector
    y_pred : output vector from softmax or others
    """
    return -np.sum(y_true * np.log(y_pred + 1e-12))


class FeedForwardNeuralNetwork:

    """
    Feed forward neural network basic class for classification.

    Example:
        
        x_train = np.array([0,1,2],[2,5,0],[4,8,2])
        y_train = np.array([0,1,1])

        #init the model
        model = (FeedForwardNeuralNetwork(X = x_train, y = y_train, loss = 'binary_cross_entropy',
                learning_rate = 0.01, epoch = 10))

        #add your layers
        model.add_hidden_layer(layer_rank=1, neurons = 64, activation = 'relu')
        model.add_hidden_layer(layer_rank=2, neurons = 12, activation = 'relu')
        model.add_output_layer(neurons = 1, activation = 'sigmoid')

        #train the model and get the mean loss per epoch
        loss = model.train()

        #predict (prob)
        y_test = np.array([1])
        x_test = np.array([4,8,3])

        pred_prob = model.predict_prob()

        #save the model
        model.save_param(model_name = 'model_example')  #create a pkl file 'model_example.pkl'

        #train the model with the saved param
        import pickle

        with open('model_example.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
        
        model.pretrain(loaded_model)

    Educational purpose only.

    """

    def __init__(self, X, y, loss: str, learning_rate: float, epoch: int):

        """
        X : pd.DataFrame or np.array. Should be the training set

        y : pd.DataFrame or np.array. Should be the training target set

        loss : only 'binary_cross_entropy' or 'cross_entropy'

        learning_rate : corresponds to the alpha of W += alpha * dW

        epoch : Number of epoch during the training phase. An epoch corresponds to passing through
        all examples.
        """
        
        if isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame):
            X = X.to_numpy()
            y = y.to_numpy()
            self.X = X.T
            self.y = y.T
        else:
            self.X = X.T
            self.y = y.T
            
        self.loss = loss
        self.learning_rate = learning_rate
        self.epoch = epoch

        self.couches = list()
        self.nb_input = self.X.shape[0]
        self.nb_previous = 0
        self.rank = 0
        self.loss_save = list()
        self.print_loss_list = list()

        # init necessary elements for forward phase and backward phase

        self.W = dict()             # dict of weights
        self.B = dict()             # dict of biases
        self.H = dict()             # dict of activation units
        self.Z = dict()             # dict of linear units
        self.Act = list()           # list of chosen activation functions
        self.Act_prime = list()     # list of chosen derivatives activation functions
        self.G = dict()             # dict of loss partial derivatives with respect to activation units
        self.dW = dict()            # dict of loss partiel derivatives with respect to weights
        self.dB = dict()

    def add_hidden_layer(self, layer_rank: int, neurons: int, activation: str):

        """
        layer_rank : rank of the hidden layer.

        neurons : numbers of hidden neurons units

        activation : only 'relu', 'sigmoid', or 'softmax'

        """

        if layer_rank == self.rank:
            raise ValueError("Please select a proper layer rank")
            return None

        if len(self.couches) == 0:
            couche = {
                "W": np.random.randn(self.nb_input, neurons)*np.sqrt(2.0 / (self.nb_input)),
                "B": np.random.randn(neurons, 1),
                "Act": activation
                }
            self.nb_previous = neurons
            self.couches.append(couche)
            self.rank = layer_rank

        else: 
            couche = {
                "W": np.random.randn(self.nb_previous, neurons)*np.sqrt(2.0 / (self.nb_previous)),
                "B": np.random.randn(neurons, 1),
                "Act": activation
            }
            self.nb_previous = neurons 
            self.couches.append(couche)
    
    def add_output_layer(self, neurons: int, activation: str):

        """
        Do not precise the layer_rank.

        neurons : numbers of output neurons units

        activation : only 'relu', 'sigmoid', or 'softmax'

        """

        if self.rank == 0:
            raise BaseException("Please add hidden layers before")
            return None
        
        couche = {
            "W": np.random.randn(self.nb_previous, neurons)*np.sqrt(2.0 / self.nb_previous),
            "B": np.random.randn(neurons, 1),
            "Act": activation
        }
        self.couches.append(couche)

        self.init_act()
    
    def init_act(self):
        # activation functions choices
        for dic in self.couches:
            l = dic["Act"] 

            if l == "relu":
                self.Act.append(relu)
                self.Act_prime.append(relu_prime)
            elif l == "sigmoid":
                self.Act.append(sigmoid)
                self.Act_prime.append(1)    # Usefull for index
            elif l == "softmax":
                self.Act.append(softmax)
                self.Act_prime.append(1)    # Usefull for index
        
        # loss function choice
        if self.loss == "binary_cross_entropy" or self.loss == binary_cross_entropy:
            self.loss = binary_cross_entropy
        elif self.loss == "cross_entropy" or self.loss == cross_entropy:
            self.loss = cross_entropy
        else:
            raise ValueError(f"The loss '{self.loss}' is not implemented yet, or doesn't exist")

    def forward_phase(self, x_train):

        nb_layers = len(self.couches)
        x_train = x_train.reshape(-1, 1)

        # init of the 1st layer
        self.W["layer1"] = self.couches[0]["W"]
        self.B["layer1"] = self.couches[0]["B"]

        self.couches[0]["W"] = self.W["layer1"]
        self.couches[0]["B"] = self.B["layer1"]

        self.Z["layer1"] = self.W["layer1"].T @ x_train + self.B["layer1"]
        self.H["layer1"] = self.Act[0](self.Z["layer1"])

        # forward the weights through the network
        for i in range(2, nb_layers + 1):
            
            layer = f"layer{i}"
            layer_prev = f"layer{i-1}"

            self.W[layer] = self.couches[i-1]["W"]
            self.B[layer] = self.couches[i-1]["B"]

            self.couches[i-1]["W"] = self.W[layer]
            self.couches[i-1]["B"] = self.B[layer]

            self.Z[layer] = self.W[layer].T @ self.H[layer_prev] + self.B[layer]
            self.H[layer] = self.Act[i-1](self.Z[layer])
            
    def backward_phase(self, x_train, y_true):

        """
        y_true : vector if loss is cross_entropy / scalar if loss is binary_cross_entropy
        """
        x_train = x_train.reshape(-1, 1)
        y_true = y_true.reshape(-1, 1)
        nb_layers = len(self.couches)
        
        layer_out = f"layer{nb_layers}"

        # save the loss value from previous forward phase
        self.loss_save.append(self.loss(y_true, self.H[layer_out])) 

        layer_out = f"layer{nb_layers}"

        # backward phase
        for i in range(nb_layers, 0, -1):

            layer = f"layer{i}"
            layer_forw = f"layer{i+1}"
            layer_backw = f"layer{i-1}"
            
            if i == nb_layers:
                delta = (self.H[layer] - y_true)
                self.dW[layer] = self.H[layer_backw] @ delta.T
                self.dB[layer] = delta

            else:
                self.G[layer] = self.W[layer_forw] @ delta 
                delta = (self.G[layer] * self.Act_prime[i-1](self.Z[layer]))
                self.dW[layer] = self.H[layer_backw] @ delta.T if i > 1 else x_train @ delta.T
                self.dB[layer] = delta
    
    def train(self):

        pred_list = list()
        true_list = list()
        
        nb_layers = len(self.couches)
        layer_out = f"layer{nb_layers}"

        for e in range(self.epoch):

            # mimic a pure stochastic gradient descent
            rand_index = np.random.permutation(self.X.shape[1])

            for i in rand_index:
                
                x_train = self.X[:, i]
                if self.loss == cross_entropy:  # case of multilabel classification
                    y_true = self.y[:, i]
                    true_cat = np.argmax(y_true)
                    true_list.append(true_cat)
                else:
                    y_true = self.y[i]          # case of binary classification
                    true_list.append(y_true)
                  
                self.forward_phase(x_train)
                self.backward_phase(x_train, y_true)

                if self.loss == cross_entropy:
                    y_pred = np.argmax(self.H[layer_out])
                    pred_list.append(y_pred)
                else:
                    if self.H[layer_out][0] > 0.5:
                        y_pred = 1
                    else:
                        y_pred = 0
                    pred_list.append(y_pred)
                    
                

                # gradient actualisation
                for j in range(nb_layers, 0, -1):
                    layer = f"layer{j}"

                    self.W[layer] = self.W[layer] - self.learning_rate * self.dW[layer]
                    self.couches[j-1]["W"] = self.W[layer]

                    self.B[layer] = self.B[layer] - self.learning_rate * self.dB[layer]
                    self.couches[j-1]["B"] = self.B[layer]

            print_loss = np.mean(self.loss_save)
            self.loss_save = list()
            self.print_loss_list.append(print_loss)

            accuracy_epoch = accuracy_score(true_list, pred_list)
            pred_list = list()
            true_list = list()

            print(f"Accuracy on epoch {e+1} : ", accuracy_epoch)
            print(f"Loss {self.loss} on epoch {e+1}: ", print_loss)
            
        return self.print_loss_list

    def predict_prob(self, x_test):
        
        pred_prob = list()
        nb_layers = len(self.couches)

        if isinstance(x_test, pd.DataFrame):
            x_test = x_test.to_numpy()
            x_test = x_test.T
        else:
            x_test = x_test.T
        
        output_layer = f"layer{nb_layers}"

        for i in range(x_test.shape[1]):
            x_it = x_test[:,i].reshape(-1,1)
            self.forward_phase(x_it)
            
            if self.loss == binary_cross_entropy:
                pred_prob.append(self.H[output_layer][0])
            else:
                pred_prob.append(self.H[output_layer])
        return pred_prob

    def save_param(self, model_name: str):

        file_name = f"{model_name}.pkl"
        with open(file_name, "wb") as file:
            pickle.dump(self.couches, file)
        print(f"Weights properly saved as {file_name}")   

    def pre_train(self, couches):

        if isinstance(couches, list) and isinstance(couches[0], dict):
            self.couches = couches
            self.init_act()
        else:
            raise TypeError("Saved params are not properly adapted. Must be 'list' of 'dict'. Use .save_param method to save properly trained models.")

