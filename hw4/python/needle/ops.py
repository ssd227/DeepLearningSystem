"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


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
                in_grad.append(init.zeros_like(value))
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
        return a + numpy.float32(self.scalar)

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
        return a * numpy.float32(self.scalar)

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
        return a.permute(permute_axes).compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION
        

def transpose(a, axes=None):
    return Transpose(axes)(a)

class Permute(TensorOp):
    def __init__(self, axes: tuple):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.permute(self.axes).compact() # todo compact优化，自动调用。不然能出一堆bug
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        new_axes = [0]*len(self.axes)
        for i in range(len(self.axes)):
            new_axes[self.axes[i]] = i
        return permute(out_grad, new_axes)   
        # axis = self.axes
        # idxs = [axis.index(i) for i in range(len(axis))]
        # return permute(out_grad, idxs)
        ### END YOUR SOLUTION

def permute(a, axes):
    return Permute(axes)(a)

class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        a.compact() # todo 待check，这里加上处理，其他地方就可以少写一点了
        # return a.reshape(self.shape).compact() # todo 要不要这么写呢
        return a.reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return reshape(out_grad, a.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)

# 注意：broadcast_to 只处理相同的维度，把1的那些轴转成其他
# 然而test case里却并不是，ndarray的实现，限制了这个op的部分功能
class BroadcastTo(TensorOp): 
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape).compact() 
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


# 支持多维度矩阵乘，但是有很多连锁bug需要处理。
# bug a@b 在下面的实现中会修改 a和b的值，导致在backword时，会产生 高维矩阵@ 高维矩阵 的情况
#
# class MatMul(TensorOp):
#     def compute(self, a, b):
#         ### BEGIN YOUR SOLUTION
        
#         # 由于ndarray只实现了二维矩阵对二维矩阵的乘法运算
#         # 预期功能：
        
#         # a,b 必须有一个元素是二维的
#         # 非二维的矩阵，都先reshape到二维
#         # 结果reshape回预期形状
        
        
#         # preprocess
#         # todo 如果ndim==1，其实还是需要对outshape进行降维度
#         print("sgd-matmul-log", a.ndim, b.ndim)
        
#         if a.ndim == 1:
#             a = a.reshape((1, a.shape[0]))
#         if b.ndim == 1:
#             b = b.reshape((b.shape[0], 1))
#         # dim a and b >=2 ---------------------------------
        
#         # CASE 1
#         if a.ndim == 2 and b.ndim == 2:
#             return a@b
        
#         # CASE 2
#         if a.ndim > 2 and b.ndim == 2:
#             # a[.X.,M,N]  b[N,P]  out[.X.,M,P]
#             assert a.shape[-1] == b.shape[0]
#             M,N = a.shape[-2:]
#             P = b.shape[-1]
#             _X_ = list(a.shape[:-2])
#             if isinstance(_X_, int):
#                 _X_ =list(_X_)
            
#             xm = array_api.prod(_X_) * M
#             return (a.reshape((xm,N)) @ b).reshape(_X_+[M]+[P])
        
#         # CASE 3
#         if a.ndim == 2 and b.ndim > 2:
#             # a[M,N] @ b[.X.,N,P] = out[.X.,M, P]
#             assert a.shape[1] == b.shape[-2]
#             M,N = a.shape
#             P = b.shape[-1]
#             _X_ = list(a.shape[:-2])
#             if isinstance(_X_, int):
#                 _X_ =list(_X_)
                
#             xp = array_api.prod(b.shape) / b.shape[-2]
            
#             pm_axes = list(range(b.ndim))
#             pm_axes[0], pm_axes[-2] = pm_axes[-2], pm_axes[0]
            
#             # permute b two [N,.X.,P]
#             b_= b.permute(pm_axes).compact() \
#                 .reshape((b.shape[-2], xp)) # [N,XP]
            
#             #a@b [M,XP]
#             out = (a@b_).reshape([M]+_X_+[P]) # [M,.X.,P]
#             return out.permute(pm_axes).compact() # [.X.,M,P]
    
#         raise ValueError("a@b not support two high-dim array matmul")
#         ### END YOUR SOLUTION

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
        # todo 待完善，不支持 tensor > 0 的直接比较
        # mask = node.inputs[0].detach() > 0
        
        mask = Tensor(NDArray(node.numpy()>0, device=node.device), requires_grad=False, device=node.device)
        return out_grad * mask
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

# help func for class Max
def calc_numpy_max_mask(axis, keepdims, npz):
    """
        使用numpy计算max mask
        
        使用numpy计算，逻辑上是正确的。但是由于是cpu计算，会拖慢整体计算速度。
        先保证逻辑上的正确性吧。
    """
    npz_max = numpy.max(npz, axis=axis, keepdims=keepdims)

    max_axes = axis
    if axis is None:
        max_axes = list(range(len(npz.shape)))
    max_shape = list(npz.shape)
    for ax in max_axes:
        max_shape[ax] = 1
                
    if keepdims == False:
        npz_max = npz_max.reshape(max_shape)

    np_all_max =  numpy.broadcast_to(npz_max, npz.shape)

    npmask =  numpy.float32(npz == np_all_max)
    # 注意维度相同的两个max值需要均分
    if axis == None:
        npmask = npmask / numpy.sum(npmask, axis=axis)
    else:
        npmask /= numpy.sum(npmask, axis=axis, keepdims=True)
    return npmask, max_shape

class Max(TensorOp):
    # 只支持None axis, 或者 axis为int
    def __init__(self, axis: Optional[tuple] = None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims
        if isinstance(axis, int):
            self.axis = (axis,)
        assert self.axis == None or len(self.axis)==1, "only support None or one int var for axis" 
    
    def compute(self, Z):
        return Z.max(axis=self.axis, keepdims=self.keepdims)
    
    def gradient(self, out_grad, node):
        Z = node.inputs[0].detach()
        npmask, max_shape = calc_numpy_max_mask(self.axis, self.keepdims, Z.numpy())
        mask_tensor = Tensor(NDArray(npmask, device=node.device), requires_grad=False, device=node.device)
        
        return mask_tensor * broadcast_to(reshape(out_grad, max_shape), Z.shape)
    
def max(a, axis=None, keepdims=False):
    return Max(axis, keepdims)(a)


"""
LogSumExp 计算梯度的一些思路

# 由于问题等价，所以反向的梯度能不能直接用变型前的公式求？？
# 答：不行，因为计算反向梯度时，调用的中间值也会存在数值溢出问题

[done-有点小问题]尝试一：使用自己推导的梯度公式(实现 max op with axes)
[看不懂不看了]尝试二：理解bfsh的那一行代码是什么意思。(不理解)
[done] 尝试三：needle重构时，把这个算子废除掉，直接用原子op堆出来比较简单。
"""
class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
        if isinstance(axes, int):
            self.axes = (axes,)

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        maxz = Z.max(axis=self.axes, keepdims=True) # NDArray 操作
        # 由于needle不能自动broadcast, 这里需要手动操作。 注意：hw2里的numpy 支持 auto broadcast操作
        sum_exp = array_api.exp(Z - maxz.broadcast_to(Z.shape)).sum(axis=self.axes, keepdims=True)
        res = array_api.log(sum_exp) + maxz
        
        # 把res在self.axes轴上的维度压缩掉。
        if self.axes != None:
            new_shape = []
            for i in range(len(res.shape)):
                if i not in self.axes: # 非压缩轴
                    new_shape.append(res.shape[i])
            res = array_api.reshape(res, new_shape)
        else:
            res = array_api.reshape(res, (1,))
        return res
        ### END YOUR SOLUTION
    
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0].detach()
        
        # help var
        npmask, max_shape = calc_numpy_max_mask(self.axes, False, Z.numpy())
        mask_tensor = Tensor(NDArray(npmask, device=node.device), requires_grad=False, device=node.device)
        sum_shape = max_shape

        # Part 1： max(Z)部分的梯度计算
        dz1 = mask_tensor * broadcast_to(reshape(out_grad, sum_shape), Z.shape)
        
        # Part 2： LogSumExp(Z-max(Z))部分的梯度计算
        zmax = max(Z, axis=self.axes, keepdims=True)
        z_zmax = Z - broadcast_to(zmax, Z.shape)
        exp_z_zmax = exp(z_zmax)
        sum_exp_z_zmax = summation(exp_z_zmax, self.axes) # todo ops.sum操作不支持keepdims, 此时已经被压缩
        
        d_log = out_grad / sum_exp_z_zmax
        d_sum = broadcast_to(reshape(d_log, sum_shape), Z.shape) # sum的梯度=按压缩轴反向展开
        d_exp = d_sum * exp_z_zmax
        
        # dz2有两部分构成
        dz = d_exp
        dzmax = -1 * mask_tensor * d_exp 
        dz2 = dz + dzmax
        
        # print("\nops.logsumexp",
        #       "\nd_log\n", d_log,
        #       "\nd_sum\n", d_sum,
        #       "\nd_exp\n", d_exp,
        #       "\ndz1\n",dz1,
        #       "\ndz\n",dz,
        #       "\ndzmax\n",dzmax,
        #       "\ndz2\n", dz2)

        return dz1 + dz2
        ### END YOUR SOLUTION

# class LogSumExp_old(TensorOp):
#     def __init__(self, axes: Optional[tuple] = None):
#         self.axes = axes
#         if isinstance(axes, int):
#             self.axes = (axes,)

#     def compute(self, Z):
#         ### BEGIN YOUR SOLUTION
#         maxz = Z.max(axis=self.axes, keepdims=True) # NDArray 操作
#         # 由于needle不能自动broadcast, 这里需要手动操作。 注意：hw2里的numpy 支持 auto broadcast操作
#         sum_exp = array_api.exp(Z - maxz.broadcast_to(Z.shape)).sum(axis=self.axes, keepdims=True)
#         res = array_api.log(sum_exp) + maxz
        
#         # 把res在self.axes轴上的维度压缩掉。
#         if self.axes != None:
#             new_shape = []
#             for i in range(len(res.shape)):
#                 if i not in self.axes: # 非压缩轴
#                     new_shape.append(res.shape[i])
#             res = array_api.reshape(res, new_shape)
#         else:
#             res = array_api.reshape(res, (1,))
#         return res
#         ### END YOUR SOLUTION

#     def gradient(self, out_grad, node):
#         ### BEGIN YOUR SOLUTION
#         Z = node.inputs[0].detach()
        
#         # broadcast_to 先整理原始轴坐标（由于needle里broadcast实现的缺陷问题）
#         # step1: 因sum操作压缩的轴
#         sum_axes = self.axes
#         if self.axes is None:
#             sum_axes = list(range(len(Z.shape)))
#         # step2: 遵循ndarray底层的思路，先reshape补全因sum操作缺失的shape维度到1，然后broadcast到输入shape
#         ori_sum_shape = list(Z.shape)
#         for ax in sum_axes:
#             ori_sum_shape[ax] = 1
        
#         Z = Z - broadcast_to(reshape(node, ori_sum_shape), Z.shape) # 看不懂这行操作的意义
        
#         exp_z = exp(Z)
#         sum_exp_z = summation(exp_z, self.axes)
        
#         grad_sum_exp = out_grad * (sum_exp_z**(-1))
#         # print(out_grad.shape, sum_exp_z.shape, grad_sum_exp.shape)
                
#         input_grad = grad_sum_exp.reshape(ori_sum_shape).broadcast_to(Z.shape) * exp_z
#         # print(Z.shape)
#         # print(exp_z.shape)
        
#         return input_grad
#         ### END YOUR SOLUTION
        
def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (1-node.detach()**2)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


# stack 要求被拼接的元素同样的维度， stack后axis是新的维度，维度+1
class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        ### BEGIN YOUR SOLUTION
        # todo 这么实现不是费老劲了。不能直接在ndarray里实现stack？？？
        arrays = [x.numpy() for x in args]
        res = numpy.stack(arrays, axis=self.axis)
        return NDArray(res,device=args[0].device)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out_grad_tuple =  split(out_grad, axis=self.axis)
        # calc ori_shape
        ori_shape= list(out_grad_tuple[0].shape)
        del ori_shape[self.axis]
        # todo 有必要reshape吗？ split会保持分割的那个维度吗，shape=1
        input_grads = [reshape(x, tuple(ori_shape)) for x in out_grad_tuple]
        return make_tuple(*input_grads)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        res = numpy.split(A.numpy(), A.shape[self.axis], axis=self.axis)
        return tuple([NDArray(x,device=A.device) for x in res])
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis).reshape(node.inputs[0].shape)
        return stack(out_grad, self.axis) # todo 这个reshape 有必要吗？
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION

def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        shape = list(a.shape)
        
        for i in self.axes:
            shape[i] *= (self.dilation+1)
        
        out_arr = array_api.empty(tuple(shape), device=a.device)
        out_arr.fill(0)
        
        # set value
        slicing = []
        for i, s in enumerate(a.shape):
            if i in self.axes:
                slicing.append(slice(None, None, self.dilation+1))
            else:
                slicing.append(slice(s))      
        out_arr[tuple(slicing)] = a
        
        return out_arr
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        shape = list(a.shape)
        
        for i in self.axes:
            shape[i] /= (self.dilation+1)
        
        # set value
        slicing = []
        for i, s in enumerate(a.shape):
            if i in self.axes:
                slicing.append(slice(None, None, self.dilation+1))
            else:
                slicing.append(slice(s))
        
        return a[tuple(slicing)].compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)

# todo
class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        # hint: im2col
        if self.padding != 0:
            axes = ( (0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)) #NHWC
            A = A.pad(axes)
        
        N,H,W,C_in = A.shape
        K,_,_,C_out = B.shape
        Ns, Hs, Ws, Cs = A.strides
        
        H_out = (H-K)//self.stride+1
        W_out = (W-K)//self.stride+1
        
        # todo 注意：这里的stride容易错，后两位不需要乘self.stride. 
        # 因为在kk内移动和原始矩阵移动没有区别，但是跳方框时会有stride的区别
        new_strides =  (Ns, Hs*self.stride, Ws*self.stride, Hs, Ws, Cs)
        new_shape =  (N, H_out, W_out, K, K, C_in)
        
        inner_dim = K * K * C_in
        inner_left_dim=N*H_out*W_out

        A = A.as_strided(shape =new_shape,
                            strides =new_strides).compact().reshape((inner_left_dim, inner_dim))
        
        B= B.compact().reshape((inner_dim, C_out))
        
        out = A @ B
        return out.compact().reshape((N, H_out, W_out, C_out))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X, W = node.inputs[:2]
        
        print("X[{}], W[{}], out_grad[{}]".format(X.shape, W.shape, out_grad.shape))
        K,_,C_in,C_out = W.shape
        
        new_padding= K - self.padding - 1
        dX= conv(out_grad.dilate((1,2),self.stride-1), W.flip((0,1)).transpose() , padding=new_padding)
        
        # print("dX[{}]".format(dX.shape))
        
        # X：[N, H, W, C_in] to [C_in, H, W, N]
        # grad：[N, H_out, W_out, C_out] to [H_out, W_out, N, C_out]
        # 
        # [C_in, H, W, N] @ [H_out, W_out, N, C_out] = [C_in, K,K,C_out]  ==> permute to [K, K, C_in, C_out]
        # dW = conv(X.permute((3,1,2,0)), out_grad.permute((1,2,0,3)), padding=self.padding).permute((1,2,0,3))
        dW = conv(X.permute((3,1,2,0)), out_grad.dilate((1,2),self.stride-1).permute((1,2,0,3)), padding=self.padding).permute((1,2,0,3))
        return dX, dW
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)



