import numpy as np
import MNIST_tools
import plot_tools

n_feature = 784
n_class = 10
n_iter = 1
cross_entropy_train = 0
cross_entropy_test = 0
loss_train = []
loss_test = []
count_train = 0
count_test = 0

def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / exps.sum()

def d_softmax(x):
    #This function has not yet been tested.
    return x.T * (1 - x)

def relu(x):
    x[x<0] = 0
    return x

def d_relu(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x.T   

def normalize(image):
    return image / (255.0 * 0.99 + 0.01)

def NeuralNetwork(n_hidden=256):
    np.random.seed(1)
    model = dict(
        W1=np.random.randn(n_feature, n_hidden),
        W2=np.random.randn(n_hidden, n_class)
    )
    return model

"""
class NeuralNetwork(object):
    def __init__(self, n_input=784, n_hidden=256, n_output=10):
        np.random.seed(1)
        self.W1=np.random.randn(n_input, n_hidden)
        self.W2=np.random.randn(n_hidden, n_output)
"""

def forward(x, model):
    # Input to hidden
    h = x @ model['W1']
    # ReLU non-linearity
    h[h < 0] = 0
    # Hidden to output
    prob = softmax(h @ model['W2'])

    return h, prob

def backward(model, xs, hs, errs):
    """xs, hs, errs contain all informations (input, hidden state, error) of all data in the minibatch"""
    # errs is the gradients of output layer for the minibatch
    dW2 = hs.T @ errs

    # Get gradient of hidden layer
    dh = errs @ model['W2'].T
    dh[hs <= 0] = 0

    dW1 = xs.T @ dh

    return dict(W1=dW1, W2=dW2)

def sgd(model, X_train, y_train, minibatch_size):
    global loss_train, cross_entropy_train, count_train
    for iter in range(n_iter):
        cross_entropy_train = 0
        print('Iteration {}'.format(iter))
        for i in range(0, X_train.shape[0], minibatch_size):
            # Get pair of (X, y) of the current minibatch/chunk
            X_train_mini = X_train[i:i + minibatch_size]
            y_train_mini = y_train[i:i + minibatch_size]
            model = sgd_step(model, X_train_mini, y_train_mini)
        cross_entropy_train /= count_train
        loss_train.append(-cross_entropy_train)
        count_train = 0
        print(loss_train)
    return model

def sgd_step(model, X_train, y_train):
    grad = get_minibatch_grad(model, X_train, y_train)
    model = model.copy()

    # Update every parameters in our networks (W1 and W2) using their gradients
    for layer in grad:
        # Learning rate: 1e-4
        model[layer] += 1e-4 * grad[layer]

    return model

def get_minibatch_grad(model, X_train, y_train):
    global cross_entropy_train, count_train
    xs, hs, errs = [], [], []

    for x, cls_idx in zip(X_train, y_train):
        h, y_pred = forward(x, model)

        # Create probability distribution of true label
        y_true = np.zeros(n_class)
        y_true[int(cls_idx)] = 1.

        # Compute the gradient of output layer
        err = y_true - y_pred
        
        cross_entropy_train += np.log(y_pred[int(cls_idx)])
        count_train += 1
        #cross_entropy += np.sum((y_true*np.log(y_pred)))
        # Accumulate the informations of minibatch
        # x: input
        # h: hidden state
        # err: gradient of output layer
        xs.append(x)
        hs.append(h)
        errs.append(err)
    
    # Backprop using the informations we get from the current minibatch
    return backward(model, np.array(xs), np.array(hs), np.array(errs))

if __name__ == '__main__':
    X_train, y_train = MNIST_tools.loadMNIST(dataset="training", path="MNIST_data")
    X_test, y_test = MNIST_tools.loadMNIST(dataset="testing", path="MNIST_data")

    X_train = normalize(X_train)
    X_test = normalize(X_test)

    minibatch_size = 32
    experiment_times = 30

    # Create placeholder to accumulate prediction accuracy
    accs = np.zeros(experiment_times)
    model = NeuralNetwork()

    for k in range(experiment_times):
        print('* Experiment time {}'.format(k+1))
        # Reset model
        # Train the model
        print('* Training Neural Network')
        model = sgd(model, X_train, y_train, minibatch_size)
        y_pred = np.zeros_like(y_test)
        print('* Testing Neural Network')
        for i, x in enumerate(X_test):
            # Predict the distribution of label
            _, prob = forward(x, model)
            cross_entropy_test += np.log(prob[y_test[i]])
            # Get label by picking the most probable one
            y = np.argmax(prob)
            y_pred[i] = y
            #if i % 100 == 0: print(y_test[i], y_pred[i], np.rint(prob))
        cross_entropy_test /= len(X_test)
        loss_test.append(-cross_entropy_test)
        print(loss_test)
        # Compare the predictions with the true labels and take the percentage
        accs[k] = (y_pred == y_test).sum() / y_test.size
        print('Mean Accuracy: {}'.format(accs[k]))
    plot_tools.plot_list(accs, title="Accuracy")
    plot_tools.plot_list(loss_test, title="Testing loss")
    plot_tools.plot_list(loss_train, title="Training loss")
    
    print('Mean Accuracy: {}, std: {}'.format(accs.mean(), accs.std()))
    
