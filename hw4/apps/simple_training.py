import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
from needle import backend_ndarray as nd
from models import *
import time

device = ndl.cpu()

### CIFAR-10 training ###

def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt:
        model.train()
    else:
        model.eval()
 
    loss_list = []
    all_N, pred_true_N = 0, 0

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
        pred_true_N +=  np.sum(label.numpy() == y_pred)
    
    # output：avg_acc, avg_loss
    return  pred_true_N/all_N, np.mean(loss_list)
    ### END YOUR SOLUTION


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    # training
    accum_loss = 0
    accum_acc = 0
    for i in range(n_epochs):
        print("epoch: ",i)
        # training
        train_avg_acc, train_avg_loss = epoch_general_cifar10(dataloader=dataloader, model=model, loss_fn=loss_fn, opt=opt)
        print("train_avg_acc:{}, train_avg_loss:{}.".format(train_avg_loss, train_avg_loss))
        accum_acc += train_avg_acc
        accum_loss += train_avg_loss
    return accum_acc/n_epochs, accum_loss/n_epochs
    ### END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION

    # evaluate
    avg_acc, avg_loss = epoch_general_cifar10(dataloader=dataloader, model=model, loss_fn=loss_fn)
    print("eval_avg_acc:{}, eval_avg_loss:{}.".format(avg_loss, avg_loss))
    
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def batch_loss(logits: nn.Tensor, y: nn.Tensor):
    r, n = logits.shape
    zy = ndl.ops.summation(logits * nn.init.one_hot(n, y, device=logits.device),axes=1)
    res = ndl.ops.summation(nn.ops.logsumexp(logits, (1,)) - zy)
    return res

### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt == None:
        model.eval()
    else:
        model.train()
    nbatch, batch_size = data.shape
    avg_loss = np.float32(0)
    avg_acc = np.float32(0)
    sum_samples = np.float32(0)
    # for i in range(nbatch - seq_len):
    for i in range(0, nbatch - 1, seq_len):
        
        batch_x, batch_y = ndl.data.get_batch(data, i, seq_len, device=device, dtype=dtype)
        sum_samples += batch_size * batch_x.shape[0]
        if opt == None:
            out, _ = model(batch_x)
            loss = loss_fn(out, batch_y)
        else:
            opt.reset_grad()
            out, _ = model(batch_x)
            loss = loss_fn(out, batch_y)
            loss.backward()
            if getattr(opt, 'clip_grad_norm', None) is not None:
                if clip is not None:
                    opt.clip_grad_norm(clip)
                else:
                    opt.clip_grad_norm()
            opt.step()
        bloss = batch_loss(out, batch_y)
        bacc = accuracy(out.numpy(), batch_y.numpy())
        avg_loss += bloss
        avg_acc += bacc
        print("batch:{} \t batch_loss{} \t batch_acc{}".format(i, bloss, bacc))
    return avg_acc / np.float32(sum_samples), avg_loss.numpy() / np.float32(sum_samples)  
    ### END YOUR SOLUTION


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
          device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for i in range(n_epochs):
        print("epoch:{}".format(i))
        avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len=seq_len, loss_fn=loss_fn(), opt=opt, clip=clip, device=device, dtype=dtype)
        # print("loss: ", avg_loss, "acc: ", avg_acc)
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len=seq_len, loss_fn=loss_fn(), opt=None, device=device, dtype=dtype)
    return avg_acc, avg_loss
    ### END YOUR SOLUTION

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
      y_hat = np.argmax(y_hat, axis=1)
    cmp = y_hat == y.astype('int')
    return np.float32(np.sum(cmp))

if __name__ == "__main__":
    ### For testing purposes
    device = ndl.cpu()
    #dataset = ndl.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=True)
    #dataloader = ndl.data.DataLoader(\
    #         dataset=dataset,
    #         batch_size=128,
    #         shuffle=True
    #         )
    #
    #model = ResNet9(device=device, dtype="float32")
    #train_cifar10(model, dataloader, n_epochs=10, optimizer=ndl.optim.Adam,
    #      lr=0.001, weight_decay=0.001)

    corpus = ndl.data.Corpus("./data/ptb")
    seq_len = 40
    batch_size = 16
    hidden_size = 100
    train_data = ndl.data.batchify(corpus.train, batch_size, device=device, dtype="float32")
    model = LanguageModel(1, len(corpus.dictionary), hidden_size, num_layers=2, device=device)
    train_ptb(model, train_data, seq_len, n_epochs=10, device=device)
