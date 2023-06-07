"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        # 可以参考pytorch-doc里的实现
        # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
        for param in self.params:
          grad = param.grad.data
          if self.weight_decay != 0:
            grad += self.weight_decay * param.data
            
          if self.momentum !=0:
            grad =  (1-self.momentum) * grad + self.momentum * self.u.get(param, 0)
          
          self.u[param] = grad

          param.data -= self.lr * ndl.Tensor(grad.data, dtype=param.dtype)
          # todo 为什么非得这么写，错一点都不行。 grad.data直接用不行，dtype不写也不通过
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for param in self.params:
            grad = param.grad.data
            if self.weight_decay != 0:
                grad += self.weight_decay * param.data
            
            nm = self.m.get(param, 0) * self.beta1 + (1-self.beta1) * grad
            nv = self.v.get(param, 0) * self.beta2 + (1-self.beta2) * (grad**2)
            self.m[param], self.v[param] = nm, nv

            m_with_bias_corr = (nm / (1 - self.beta1 ** self.t))
            v_with_bias_corr = (nv / (1 - self.beta2 ** self.t))

            update = self.lr * (m_with_bias_corr) / (v_with_bias_corr ** 0.5 + self.eps)
            param.data -= ndl.Tensor(update, dtype=param.dtype)
        ### END YOUR SOLUTION
