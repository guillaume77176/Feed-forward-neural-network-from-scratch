# Feed forward neural network from scratch.

This implementation is for educational purposes only. It implements a dense neural network in a basic way. The code was written to use simple types to traverse the directed acyclic graph, in particular dictionaries and lists. The code is therefore not optimal in terms of execution time.

A purely stochastic gradient descent was chosen for simplicity. Consequently, the network is trained example by example for each epoch.

You will find an example on the MNIST digit data set in the `Mnist_example.ipynb` file.

To test by hand the model, you can write your own digits and observe if it correctly predicts ! https://yourdigit.streamlit.app/. You can increase the epoch in order to have better results.

### Import the project
```bash
git clone https://github.com/guillaume77176/Feed-forward-neural-network-from-scratch.git
```

```bash
cd Feed-forward-neural-network-from-scratch
```

```bash
pip install -r requirements.txt
```

### Project structure
```
├── src/
│   └── EasyNN.py
├── mnist_ffnn.pkl
├── Mnist_example.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

### Example

``` python
from src.EasyNN import FeedForwardNeuralNetwork

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
x_test = pd.DataFrame(x_test)

pred_prob = model.predict_prob()

#save the model
model.save_param(model_name = 'model_example')  #create a pkl file 'model_example.pkl'
```

``` python
#train the model from saved params

with open('model_example.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

model.pre_train(loaded_model)
```

### Author 
guillaume77176
