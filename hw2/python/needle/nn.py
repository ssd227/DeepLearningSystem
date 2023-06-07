"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
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
    def __init__(self, 
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype="float32"
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


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for m in self.modules:
          x = m(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        lse = ops.logsumexp(logits, axes=1)
        one_hot = init.one_hot(logits.shape[-1], y, device=logits.device)
        zy = ops.summation(logits * one_hot, axes=1)
        batch_size = logits.shape[0]
        return ops.summation(lse-zy)/batch_size
        ### END YOUR SOLUTION



# todo 这里的dim全指的是channel数，和nrom的方向一点关系没有？？
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



