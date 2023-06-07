import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    sub_net = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(dim=hidden_dim),
        nn.ReLU(),
        nn.Dropout(p=drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim=dim),)

    return nn.Sequential(nn.Residual(sub_net),
              nn.ReLU(),)
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    linear_blocks = [nn.Linear(in_features=dim, out_features=hidden_dim), nn.ReLU()]
    res_blocks = [
        ResidualBlock(dim=hidden_dim, hidden_dim = hidden_dim//2, norm=norm, drop_prob=drop_prob)
        for _ in range(num_blocks)]

    return nn.Sequential(
            *linear_blocks,
            *res_blocks,
            nn.Linear(in_features=hidden_dim, out_features=num_classes),)
    ### END YOUR SOLUTION




def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt:
        model.train()
    else:
        model.eval()
 
    loss_list = []
    all_N, error_N = 0, 0

    for imgs, label in dataloader:
        X = imgs.reshape((imgs.shape[0],-1))
        y = model(X)

        loss = nn.SoftmaxLoss()(y, label)
        loss_list.append(loss.numpy())

        if opt:
            opt.reset_grad()
            loss.backward()
            opt.step()

        all_N += X.shape[0]
        y_pred = np.argmax(y.numpy(), axis=1)
        error_N +=  np.sum(label.numpy() != y_pred)
    
    # output：error_rate, avg_loss
    return  error_N/all_N, np.mean(loss_list)
    ### END YOUR SOLUTION



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    mnist_train_dataset = ndl.data.MNISTDataset("data/train-images-idx3-ubyte.gz",
                            "data/train-labels-idx1-ubyte.gz")
    train_dataloader = ndl.data.DataLoader(dataset=mnist_train_dataset, batch_size=batch_size, shuffle=True)

    mnist_test_dataset = ndl.data.MNISTDataset("data/t10k-images-idx3-ubyte.gz",
                            "data/t10k-labels-idx1-ubyte.gz")
    test_dataloader = ndl.data.DataLoader(dataset=mnist_test_dataset, batch_size=batch_size, shuffle=False)
    

    resnet = MLPResNet(28*28, hidden_dim)
    
    opt = optimizer(resnet.parameters(), lr=lr, weight_decay=weight_decay)

    # training
    for i in range(epochs):
        print("epoch: ",i)
        # training
        train_err_rate, train_loss = epoch(dataloader=train_dataloader, model=resnet, opt=opt)
        test_err_rate, test_loss = epoch(dataloader=test_dataloader, model=resnet)

        print("train_acc:{}, train_loss:{}, test_acc:{}, test_loss:{}".format(1-train_err_rate, train_loss, 1-test_err_rate, test_loss))

    # training accuracy, training loss, test accuracy, test loss computed in the last epoch of training.
    return 1-train_err_rate, train_loss, 1-test_err_rate, test_loss
    ### END YOUR SOLUTION



if __name__ == "__main__":
    train_mnist(data_dir="../data")
