"""The module.
"""
from typing import List
# from needle.autograd import Tensor
from needle.autograd import Tensor, NDArray
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
          init.kaiming_uniform(
            in_features, out_features,
            nonlinearity="relu",
            device=device, dtype=dtype, requires_grad=True))

        self.bias = None
        if bias:
          self.bias = Parameter(
              init.kaiming_uniform(
                out_features, 1,
                device=device, dtype=dtype, requires_grad=True)
              .reshape((1, out_features)))
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        h = X @ self.weight
        return h + self.bias.broadcast_to(h.shape) if self.bias else h
        # return h + self.bias.broadcast_to((X.shape[0], self.out_features)) if self.bias else h
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        if len(X.shape) ==1:
            return X.reshape(X.shape[0],1)
        if len(X.shape) == 2:
            return X
        
        # need flatten, calc new shape [m,n]
        m, n = X.shape[0], 1
        for i in X.shape[1:]:
          n *= i
        return X.reshape((m,n))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.tanh(x)
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return (1+ops.exp(-x)) ** (-1)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for f in self.modules:
          x = f(x)
        return x
        ### END YOUR SOLUTION

class LogSumExp(Module):
    """
        数值稳定版本的nn.LogSumExp
        
        直接用算子堆出来的模块，不用手动推导公式
            使用新增的Max算子来保证数值稳定性
    """
    def forward(self, Z:Tensor, axis=None, keepdims=False):
        max_Z = ops.max(Z, axis=axis, keepdims=True)
        max_Z_broadcast = ops.broadcast_to(max_Z, Z.shape)

        return ops.log(ops.summation(ops.exp(Z-max_Z_broadcast), axes=axis)) + ops.max(Z, axis=axis)

class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        lse = ops.logsumexp(logits, axes=1)
        # lse = LogSumExp().forward(logits, axis=1) # 试一试新的模块写法
        one_hot = init.one_hot(logits.shape[-1], y, device=logits.device) #todo 注意：device
        zy = ops.summation(logits * one_hot, axes=1)

        batch_size = logits.shape[0]
        return ops.summation(lse-zy)/batch_size
        ### END YOUR SOLUTION

# 这里的dim全指的是channel数，和nrom的方向一点关系没有
class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
                  init.ones(1, dim,device=device, dtype=dtype))
        self.bias = Parameter(
                init.zeros(1, dim,device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        M,N = x.shape # x[M*N]
        assert N == self.dim
        
        if self.training: # train()
          ex = (x.sum(axes=0)/M).reshape((1,N)) # ex:[1*N]
          ex_bc = ex.broadcast_to(x.shape) # ex_bc:[M*N]

          varx = (((x-ex_bc)**2).sum(axes=0)/M).reshape((1,N)) # varx:[1*N]
          varx_bc = varx.broadcast_to(x.shape) # varx_bc:[M*N]
        
          # 不支持 todo 这就很无语了，底层ndarray 不支持(N,) (N,)的加和操作？？？
          # self.running_mean = (1-self.momentum) * self.running_mean\
          #             + (self.momentum * ex).reshape((N,)).detach() # [N]
          # self.running_var = (1-self.momentum) * self.running_var.reshape((1,N)) \
          #             + (self.momentum * varx).reshape((N,)).detach() # [N]
          run_mean = self.momentum * ex + (1-self.momentum) * ops.reshape(self.running_mean, (1,N))
          self.running_mean = ops.reshape(run_mean, (N,)).detach()
          run_var = self.momentum * varx + (1-self.momentum) * ops.reshape(self.running_var, (1,N))
          self.running_var = ops.reshape(run_var, (N,)).detach()

          h = (x-ex_bc) / (varx_bc + self.eps)**0.5 # h:[M*N]
          return self.weight.broadcast_to(x.shape) * h + self.bias.broadcast_to(x.shape)
        
        # eval()
        ex_rbc = self.running_mean.broadcast_to(x.shape)
        varx_rbc = self.running_var.broadcast_to(x.shape)
        return (x-ex_rbc) / (varx_rbc + self.eps)**0.5 # h:[M*N]
        ### END YOUR SOLUTION




class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(1, dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        M,N = x.shape  # x[M*N]
        assert N == self.dim
        
        ex = (x.sum(axes=1)/N).reshape((M,1)) # ex:[M*1]
        ex_bc = ex.broadcast_to(shape=x.shape) # ex_bc:[M*N]

        varx = (((x-ex_bc)**2).sum(axes=1)/N).reshape((M,1)) # varx:[M*1]
        varx_bc = varx.broadcast_to(shape=x.shape) # varx_bc:[M*N]

        h = (x-ex_bc) / (varx_bc + self.eps)**0.5 # h:[M*N]
        return self.weight.broadcast_to(x.shape) * h + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION
    
    # todo 这里的实现是真无语，有必要把这些统计op都加到计算图里面吗？？
    # def forward(self, x: Tensor) -> Tensor:
    #     ### BEGIN YOUR SOLUTION
    #     x_data = x.detach().numpy()
    #     ex = x_data.mean(axis=1, keepdims=True)
    #     varx = ((x_data-ex)**2).mean(axis=1,keepdims=True)
    #     print(type(ex), type(varx))
    #     h = (x-ex) / (varx + self.eps)**0.5
    #     return self.weight.broadcast_to(x.shape) * h + self.bias.broadcast_to(x.shape)
    #     ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = init.randb(*x.shape, p= 1-self.p, device=x.device) / (1 - self.p) # todo 注意这里的device
            return x * mask
        return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x)+x
        ### END YOUR SOLUTION

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.weight = Parameter(
                        init.kaiming_uniform(
                            kernel_size**2 * in_channels,
                            kernel_size**2 * out_channels,
                            shape= (kernel_size, kernel_size, in_channels, out_channels),
                            device=device,
                            dtype=dtype,
                        )) # [K, K, C_in, C_out]
        
        self.bias = Parameter(
                        init.rand(
                            out_channels,
                            low= -1.0/(in_channels*kernel_size**2)**0.5,
                            high= 1.0/(in_channels*kernel_size**2)**0.5,
                            device=device,
                            dtype=dtype,
                            )) # [C_out,]
        
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        ## NCHW => NHWC
        x = ops.permute(x, (0,2,3,1))
        
        N,H,W,C_in, = x.shape
        K,_,_,C_out,= self.weight.shape
        
        ## calc padding num to keep the same (H,W)
        padding = (K - 1) // 2  # todo padding = ((W-1)*self.stride+K-W)//2 todo无语，多stride就不要对齐了？？
        ## conv
        x_conv = ops.conv(x, self.weight, self.stride, padding=padding)
        ## NHWC => NCHW
        x_conv = ops.permute(x_conv, (0,3,1,2))
        ## x_conv.shape (N, C_out, H_out, W_out, )
        return x_conv + self.bias.reshape((1,C_out, 1, 1)).broadcast_to(x_conv.shape) # 手动broadcast todo 待优化
        ### END YOUR SOLUTION


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.hidden_size = hidden_size
        self.bias = bias
        
        if nonlinearity == "relu":
            self.activate_f = ReLU()
        elif nonlinearity == "tanh":
            self.activate_f = Tanh()
        else:
            self.activate_f = Tanh()
        
        # init weight
        k = 1/ hidden_size
        bound = k**0.5
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-bound,high=bound, device=device))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-bound,high=bound, device=device))
        if bias:
            self.bias_ih = Parameter(init.rand(hidden_size, low=-bound,high=bound, device=device))
            self.bias_hh = Parameter(init.rand(hidden_size, low=-bound,high=bound, device=device))
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            h = init.zeros(X.shape[0], self.hidden_size, device=X.device, dtype=X.dtype)
        xw = X @ self.W_ih
        hw = h @ self.W_hh
        if self.bias:
            bi = self.bias_ih.reshape((1, self.hidden_size)).broadcast_to(xw.shape)
            bh = self.bias_hh.reshape((1, self.hidden_size)).broadcast_to(hw.shape)
            ht = self.activate_f(xw+bi+hw+bh)
            return ht
        # no bias
        ht = self.activate_f(xw+hw)
        return ht
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn_cells = []
        for i in range(num_layers):
            if i == 0:
                self.rnn_cells.append(RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype))
            else:
                self.rnn_cells.append(RNNCell(hidden_size, hidden_size, bias, nonlinearity, device, dtype))
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        T, B, C_in = X.shape # [Time_seq, Batch, C_in]
        
        if h0 is None:
            h0 = init.zeros(self.num_layers, B, self.hidden_size, device=X.device, dtype=X.dtype)
        
        hs = ops.split(h0, axis=0) # h[num_layers]
        Xs = ops.split(X, axis=0) # Xs[T]
        
        output = [] # 时间维度，收集 h_highlayer_t， 共T个。
        for t in range(T):
            hs_tmp = []     # 每个时间t下，收集一次 h_layer_i, 共num_layer个
            for ly in range(self.num_layers):
                if ly==0:
                    h = self.rnn_cells[ly](Xs[t].reshape((B,C_in)), hs[ly].reshape((B, self.hidden_size)))
                else:
                    h = self.rnn_cells[ly](hs_tmp[-1].reshape((B, self.hidden_size)), hs[ly].reshape((B, self.hidden_size)))
                hs_tmp.append(h)
                
            output.append(hs_tmp[-1]) # 收集顶层rnn_cell的输出h
            hs = tuple(hs_tmp) # update hidden state at t-1， as input for next time t
        
        return ops.stack(output, axis=0), ops.stack(hs, axis=0)
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.hidden_size = hidden_size
        self.bias = bias
        
        # init weight
        k = 1/ hidden_size
        bound = k**0.5
        self.W_ih = Parameter(init.rand(input_size, 4*hidden_size, low=-bound,high=bound, device=device))
        self.W_hh = Parameter(init.rand(hidden_size, 4*hidden_size, low=-bound,high=bound, device=device))
        if bias:
            self.bias_ih = Parameter(init.rand(4*hidden_size, low=-bound,high=bound, device=device))
            self.bias_hh = Parameter(init.rand(4*hidden_size, low=-bound,high=bound, device=device))
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        B,C_in = X.shape
        C_out = self.hidden_size
        
        if h is None:
            h = (init.zeros(B, C_out, device=X.device, dtype=X.dtype),
                 init.zeros(B, C_out, device=X.device, dtype=X.dtype))
        h0, c0 = h
        
        xw = X @ self.W_ih
        hw = h0 @ self.W_hh
        
        if self.bias:
            bi = self.bias_ih.reshape((1, 4*C_out)).broadcast_to(xw.shape)
            bh = self.bias_hh.reshape((1, 4*C_out)).broadcast_to(hw.shape)
            xw += bi
            hw += bh
        
        i,f,g,o = [x.reshape((B, C_out)) for x in ops.split((xw+hw).reshape((B,4,C_out)), axis=1)]
        
        i= Sigmoid()(i) # input gate
        f= Sigmoid()(f) # forget gate
        o= Sigmoid()(o) # output gate
        g = Tanh()(g) # current generated value
        
        c_ = f * c0 + i*g
        h_ = o * Tanh()(c_)
        
        return h_, c_
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        # self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        
        self.lstm_cells = []
        for i in range(num_layers):
            if i==0:
                self.lstm_cells.append(LSTMCell(input_size, hidden_size,bias=bias,device=device,dtype=dtype))
            else:
                self.lstm_cells.append(LSTMCell(hidden_size, hidden_size,bias=bias,device=device,dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        T, B, C_in = X.shape # [Time_seq, Batch, C_in]
        C_out = self.hidden_size
        
        if h is None:
            h = (init.zeros(self.num_layers, B, C_out, device=X.device, dtype=X.dtype),
                    init.zeros(self.num_layers, B, C_out, device=X.device, dtype=X.dtype))
        h0, c0 = h
        hs = ops.split(h0, axis=0) # h[num_layers]
        cs =  ops.split(c0, axis=0) # h[num_layers]
        Xs = ops.split(X, axis=0) # Xs[T]
        
        
        output = [] # 时间维度，收集 h_highlayer_t， 共T个。
        for t in range(T):
            hs_tmp, cs_tmp = [], []    # 每个时间t下，收集一次 h_layer_i, 共num_layer个
            for ly in range(self.num_layers):
                if ly==0:
                    h,c = self.lstm_cells[ly](Xs[t].reshape((B,C_in)), 
                                              (hs[ly].reshape((B, C_out)),  cs[ly].reshape((B, C_out)), )
                                              )
                else:
                    h, c = self.lstm_cells[ly](hs_tmp[-1].reshape((B, C_out)),
                                              (hs[ly].reshape((B, C_out)),  cs[ly].reshape((B, C_out)), )
                                              )
                hs_tmp.append(h)
                cs_tmp.append(c)
                
            output.append(hs_tmp[-1]) # 收集顶层rnn_cell的输出h
            hs = tuple(hs_tmp) # update hidden state at t-1， as input for next time t
            cs = tuple(cs_tmp)
        
        return ops.stack(output, axis=0), (ops.stack(hs, axis=0),ops.stack(cs, axis=0))
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.dtype = dtype
        
        self.weight = Parameter(
            init.randn(num_embeddings, embedding_dim, mean=0,std=1 ,device=device,dtype=dtype)
        )
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        T,B = x.shape
        x_np = x.numpy().astype(int)# [T, B]
        x_oh = np.eye(self.num_embeddings)[x_np] # [T, B, self.num_embeddings]
        assert x_oh.shape == (T, B, self.num_embeddings)        
        x_one_hot = Tensor(x_oh, device=x.device, dtype=x.dtype, requires_grad=False)
        
        # todo 居然不支持多维矩阵的普通乘法，那还搞毛线啊，更新ops里的操作
        # print("sgd-emb-log, a[{}] @ b[{}]".format(x_one_hot.shape, self.weight.shape) )
        
        output = x_one_hot.reshape((T*B, self.num_embeddings)) @ self.weight
        
        output = output.reshape((T, B, self.embedding_dim))
        assert output.shape == (T, B, self.embedding_dim)
        return output
        ### END YOUR SOLUTION
