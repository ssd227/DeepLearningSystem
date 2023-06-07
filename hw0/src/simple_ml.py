import struct
import numpy as np
import gzip
import math

try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x + y
    ### END YOUR CODE


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    with gzip.open(label_filename, 'rb') as label_file:
        magic, num_labels = struct.unpack(">II", label_file.read(8))
        labels = np.frombuffer(label_file.read(), dtype=np.uint8)

    with gzip.open(image_filename, 'rb') as image_file:
        magic, num_images, rows, cols = struct.unpack(">IIII", image_file.read(16))
        images = np.frombuffer(image_file.read(), dtype=np.uint8).reshape(num_images, rows*cols).astype(np.float32)
        # normlize images
        min_pv, max_pv = np.min(images) , np.max(images)
        images_normalized = (images - min_pv) / (max_pv - min_pv)
    
    return images_normalized, labels
    ### END YOUR CODE


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.int8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    # print(Z.shape)
    # print(np.sum(np.exp(Z),axis=1,keepdims=True).shape)
    # print(np.arange(Z.shape[0])[:, np.newaxis])
    # print(Z[np.arange(Z.shape[0]), y][:, np.newaxis].shape)
    
    logsum = np.log(np.sum(np.exp(Z),axis=1,keepdims=True))
    hy = Z[np.arange(Z.shape[0]), y][:, np.newaxis]

    # print("sgd-log", logsum.shape, hy.shape)
    return np.mean(logsum-hy)
    ### END YOUR CODE


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    n = X.shape[0]
    # for i in range(0, math.ceil(n/batch)):
    for i in range(0, (n+1)//batch):
      # print("batch: ",i)
      sid = 0 + i*batch
      eid = min(sid+batch, n)

      # batch data
      b_X = X[sid:eid]  # batch train data
      b_y = y[sid:eid]  # batch label
      b_m = b_X.shape[0] # num of batch samples
      
      # logits
      b_xt = np.exp(b_X @ theta)
      # logits with softmax
      b_xt_norm = np.sum(b_xt, axis=1, keepdims=True)
      b_Z = b_xt / b_xt_norm
      
      # label to one-hot
      b_Iy = np.zeros(b_Z.shape)
      b_Iy[np.arange(b_Z.shape[0]), b_y] = 1
      
      # grad with theta
      Delta = b_X.T @ (b_Z- b_Iy) / b_m
      # print(theta.shape, Delta.shape)
      # print(b_xt.shape, b_xt_norm.shape, b_Z.shape, b_Iy.shape)
      
      # update model parameter [theta]
      theta -= lr * Delta
    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    n = X.shape[0]
    # for i in range(0, math.ceil(n/batch)):
    for i in range(0, (n+1)//batch): 
      # print("batch: ",i)
      sid = 0 + i*batch
      eid = min(sid+batch, n)

      # batch data
      b_X = X[sid:eid]  # batch train data
      b_y = y[sid:eid]  # batch label
      b_m = b_X.shape[0] # num of batch samples
      
      b_Z1 = relu(b_X @ W1)

      ez1w2 = np.exp(b_Z1 @ W2)
      ez1w2_norm = np.sum(ez1w2, axis=1, keepdims=True)

      # label to one-hot
      b_Iy = np.zeros(ez1w2.shape)
      b_Iy[np.arange(ez1w2.shape[0]), b_y] = 1

      # print(b_X.shape, b_y.shape, W1.shape, W2.shape)
      # print(b_Z1.shape, ez1w2.shape, ez1w2_norm.shape, b_Iy.shape)

      b_G2 = ez1w2 / ez1w2_norm - b_Iy

      relu_mask = np.zeros(b_Z1.shape)
      relu_mask[b_Z1>0] = 1

      b_G1 = (b_G2 @ W2.T) * relu_mask

      DW1 = b_X.T @ b_G1 / b_m
      DW2 = b_Z1.T @ b_G2 / b_m

      # update theta
      W1 -= lr  * DW1
      W2 -= lr  * DW2
    ### END YOUR CODE

def relu(x):
    return np.maximum(0, x)

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")

    # print("sgd-log")
    
    # Z = X_tr @ theta
    # y = y_tr
    # logsum = np.log(np.sum(np.exp(Z),axis=1,keepdims=True))
    # hy = Z[np.arange(Z.shape[0]), y][:, np.newaxis]

    # # print(Z.shape)
    # # print(np.sum(np.exp(Z),axis=1,keepdims=True).shape)
    # # print(np.arange(Z.shape[0])[:, np.newaxis])
    # # print(Z[np.arange(Z.shape[0]), y][:, np.newaxis].shape)
    # print("sgd-log", logsum.shape, hy.shape)

    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
