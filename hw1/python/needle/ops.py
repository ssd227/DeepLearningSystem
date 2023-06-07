"""Operator implementations."""

from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks
import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
         return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * power_scalar(node.inputs[0], self.scalar-1) * self.scalar
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = [x.detach() for x in node.inputs] # 只使inputs的历史值，不搅到计算图里。
        return out_grad / b, -out_grad * a * (b**-2)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
         return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if axes is None:
          self.axes = (-2,-1) # 默认情况交换最后两个轴 exchange last two axis
        else:
          self.axes = axes 
        assert len(self.axes) == 2 # 只交换两个轴
        #todo check ndarray 里对 -1 index的容忍度

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        permute_axes = list(range(a.ndim))
        # swap two transpose axes
        permute_axes[self.axes[0]], permute_axes[self.axes[1]] = permute_axes[self.axes[1]], permute_axes[self.axes[0]]
        return a.transpose(permute_axes) # numpy 里的transpose可以兼容多个轴的处理
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return reshape(out_grad, a.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


# todo 注意：broadcast_to 只处理相同的维度，把1的那些轴转成其他
# 然而test case里却并不是，ndarray的实现，限制了这个op的部分功能
class BroadcastTo(TensorOp): 
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)
        # todo 注意这里的compact，由于新矩阵必然和老矩阵不一致，
        # 所以需要new新的临时变量, 但是backward怎么说
        
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        ndim_a = len(a.shape)
        ndim_out= len(self.shape)
        
        # print("sgd-log", a.shape, self.shape, out_grad.shape)
        # sgd-log (1,) (3, 3, 3) 不讲武德啊，居然需要处理这种case。
        # 那底层的ndarray为什么要求 reshape 给缺失的axis补上1
        
        # 反序匹配
        sum_axes = []
        for i in range(ndim_a): # ndim_a是相对较小的那一个
            id_a = ndim_a-1 -i
            id_out = ndim_out-1 -i
            if a.shape[id_a] != self.shape[id_out] and a.shape[id_a]==1:
                sum_axes.append(id_out)
        sum_axes = [id for id in range(ndim_out - ndim_a)] + sum_axes[::-1] # ndim_out剩下的轴，直接补上

        return summation(out_grad, tuple(sum_axes)).reshape(a.shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
        if isinstance(axes, int):
          self.axes = (axes,)
        
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.sum(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        
        # 因sum操作压缩的轴
        sum_axes = self.axes
        if self.axes is None:
          sum_axes = list(range(len(a.shape)))
          
        # 遵循ndarray底层的思路，先reshape补全因sum操作缺失的shape维度到1，然后broadcast到输入shape
        out_ori_shape = list(a.shape)
        for ax in sum_axes:
          out_ori_shape[ax] = 1
          
        return broadcast_to(reshape(out_grad, out_ori_shape), a.shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a@b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X, W = node.inputs
        dX = out_grad @ W.transpose()
        dW = X.transpose() @ out_grad
        # 判断是否需要对梯度sum降维
        if len(dW.shape) > len(W.shape):
          dW = summation(dW, tuple(range(len(dW.shape)-len(W.shape))))
        if len(dX.shape) > len(X.shape):
          dX = summation(dX, tuple(range(len(dX.shape)-len(X.shape))))
        
        return dX, dW
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0].detach()
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * node.detach()
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


# TODO
class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        mask = Tensor((node.inputs[0].cached_data > 0).astype(array_api.float32))
        return out_grad * mask
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

