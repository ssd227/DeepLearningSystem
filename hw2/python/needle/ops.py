"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
import numpy as array_api


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


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


# todo 这ops没必要自己求导，直接用算子堆出一个nn模块不行吗？？（需要额外实现一个max算子）
# todo: 不用算子实现，直接再nn里堆积木，需要添加max、argsmax 两个算子方便反向求梯度。
class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
        if isinstance(axes, int):
            self.axes = (axes,)
        

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        # print("z shape, axes", Z.shape, self.axes)
        maxz = array_api.max(Z, axis=self.axes, keepdims=True)
        # print('maxz', maxz.shape)
        sum_exp = array_api.sum(array_api.exp(Z - maxz), axis=self.axes, keepdims=True)
        # print('sum_exp', sum_exp.shape)
        res = array_api.log(sum_exp) + maxz 
        # print('res', res.shape)
        
        # 对res在self.axes轴上的维度压缩掉。
        if self.axes != None:
            new_shape = []
            for x in res.shape:
                if x!=1:
                    new_shape.append(x)
            res = array_api.reshape(res, new_shape)
        else:
            res = array_api.reshape(res, (1,))
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0].detach()
        
        # 由于问题等价，所以反向的梯度能不能直接用变型前的公式求？？
        # 答：不行，因为计算反向梯度时，调用的中间值也会存在数值溢出问题
        
        # broadcast_to 先整理原始轴坐标（由于needle里broadcast实现的缺陷问题）
        # step1: 因sum操作压缩的轴
        sum_axes = self.axes
        if self.axes is None:
            sum_axes = list(range(len(Z.shape)))
        # step2: 遵循ndarray底层的思路，先reshape补全因sum操作缺失的shape维度到1，然后broadcast到输入shape
        ori_sum_shape = list(Z.shape)
        for ax in sum_axes:
            ori_sum_shape[ax] = 1
        
        # 先凑合用一下数值溢出版本，通过测试。(尝试失败，直接计算数值就溢出了，必须使用下面这行)
        Z = Z - broadcast_to(reshape(node, ori_sum_shape), Z.shape) # ！！！todo！！！还是不太懂这行操作的意义
        
        exp_z = exp(Z)
        sum_exp_z = summation(exp_z, self.axes)
        
        grad_sum_exp = out_grad * (sum_exp_z**(-1))
        # print(out_grad.shape, sum_exp_z.shape, grad_sum_exp.shape)
                
        input_grad = grad_sum_exp.reshape(ori_sum_shape).broadcast_to(Z.shape) * exp_z
        # print(Z.shape)
        # print(exp_z.shape)
        
        return input_grad 
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


# todo 有点难实现
class Max(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
        if isinstance(axes, int):
            self.axes = (axes,)
        
    def compute(self, a):
        return array_api.max(a, axis=self.axes)
        
    def gradient(self, out_grad, node):
        a = node.inputs[0].detach()
        # 找到原始maxz位置，把对应的梯度赋值，其他位置保持0
        max(a, self.axes).broadcast_to(self.axes)
        
        # 初始化a.shape的0 tensor，并把对应max位置的值的赋值为out_grad
        # todo
        pass
    
def max(a, axes=None):
    return Max(axes=axes)(a)
