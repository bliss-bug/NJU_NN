from abc import ABC, abstractmethod
from collections import defaultdict
from functools import reduce
from typing import Dict, List, Optional, Tuple
import numpy as np

NDArray = np.ndarray
TENSOR_COUNTER = 0


class Op(ABC):
    @abstractmethod
    def compute(self, *args: Tuple[NDArray]) -> NDArray:
    # 前向计算. 参数args是由NDArray组成的序列Tuple，输出计算的结果NDArray
        pass
    @abstractmethod
    def gradient(self, out_grad: "Value", node: "Value") -> Tuple["Value"]:
    # 后向求导. 计算每个输入变量对应的局部伴随值(partial adjoint)
    # 参数out_grad是输出变量对应的伴随值，node是计算操作所在的计算图节点
    # 为方便编程，输出总是一个序列Tuple
        pass



class TensorOp(Op):
    # 继承计算操作类，实现张量特有的计算
    def __call__(self, *args):
        return Tensor.make_from_op(self, args)
    


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b
    
    def gradient(self, out_grad: "Tensor", node: "Tensor"):
        return out_grad, out_grad
    

def add(a, b):
    return EWiseAdd()(a, b)
    


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar
    
    def gradient(self, out_grad: "Tensor", node: "Tensor"):
        return (out_grad,)
    

def add_scalar(a, scalar):
    return AddScalar(scalar)(a)
    


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: "Tensor", node: "Tensor"):
        a, b = node.inputs
        return out_grad * b, out_grad * a


def multiply(a, b):
    return EWiseMul()(a, b)



class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: "Tensor", node: "Tensor"):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)



class PowerScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return np.power(a, self.scalar)

    def gradient(self, out_grad: "Tensor", node: "Tensor"):
        return (out_grad * self.scalar * node.inputs[0]**(self.scalar-1),)
    

def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)



class EWiseDiv(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a / b
    
    def gradient(self, out_grad: "Tensor", node: "Tensor"):
        a, b = node.inputs
        return out_grad / b, out_grad * a * (-1) / (b * b)
    

def divide(a, b):
    return EWiseDiv()(a, b)



class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a / self.scalar

    def gradient(self, out_grad: "Tensor", node: "Tensor"):
        return (out_grad / self.scalar,)


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)



class Transpose(TensorOp):
    def __init__(self, axis):
        self.axis = axis

    def compute(self, a: NDArray):
        if self.axis:
            return np.swapaxes(a, self.axis[0], self.axis[1])
        else:
            return np.swapaxes(a, -1, -2)
    
    def gradient(self, out_grad: "Tensor", node: "Tensor"):
        return (Transpose(self.axis)(out_grad),)
    

def transpose(a, dim=None):
    return Transpose(dim)(a)

    

class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a: NDArray):
        return np.reshape(a, self.shape)

    def gradient(self, out_grad: "Tensor", node: "Tensor"):
        return (Reshape(node.inputs[0].shape)(out_grad),)
    

def reshape(a, shape):
    return Reshape(shape)(a)

    

class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a: NDArray):
        return np.broadcast_to(a, self.shape)

    def gradient(self, out_grad: "Tensor", node: "Tensor"):
        in_shape = node.inputs[0].shape

        expand_num = len(self.shape)-len(in_shape)
        expand_dims = [i for i in range(expand_num)] # 扩张的维度
        for i in range(len(in_shape)):
            if in_shape[i] != self.shape[i+expand_num]:
                expand_dims.append(i+expand_num)
        
        return (out_grad.sum(tuple(expand_dims)).reshape(in_shape),)
    

def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)
    


class Summation(TensorOp):
    def __init__(self, axis: Optional[tuple] = None):
        self.axis = axis

    def compute(self, a: NDArray):
        return np.sum(a, self.axis)

    def gradient(self, out_grad: "Tensor", node: "Tensor"):
        new_shape = list(node.inputs[0].shape)
        if self.axis is not None:
            for axis in self.axis:
                new_shape[axis] = 1
        else:
            for axis in range(len(new_shape)):
                new_shape[axis] = 1
        return (out_grad.reshape(tuple(new_shape)).broadcast_to(node.inputs[0].shape),)


def summation(a, dim=None):
    return Summation(dim)(a)



class MatMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a @ b

    def gradient(self, out_grad: "Tensor", node: "Tensor"):
        a, b = node.inputs
        lgrad, rgrad = out_grad @ b.transpose(), a.transpose() @ out_grad
        if len(a.shape) < len(lgrad.shape):
            lgrad = lgrad.sum(tuple([i for i in range(len(lgrad.shape) - len(a.shape))]))
        if len(b.shape) < len(rgrad.shape):
            rgrad = rgrad.sum(tuple([i for i in range(len(rgrad.shape) - len(b.shape))]))
        return lgrad, rgrad


def matmul(a, b):
    return MatMul()(a, b)



class Negate(TensorOp):
    def compute(self, a: NDArray):
        return -a

    def gradient(self, out_grad: "Tensor", node: "Tensor"):
        return (-out_grad,)
    

def negate(a):
    return Negate()(a)



class Log(TensorOp):
    def compute(self, a: NDArray):
        return np.log(a)

    def gradient(self, out_grad: "Tensor", node: "Tensor"):
        return (out_grad / node.inputs[0],)


def log(a):
    return Log()(a)



class Exp(TensorOp):
    def compute(self, a: NDArray):
        return np.exp(a)

    def gradient(self, out_grad: "Tensor", node: "Tensor"):
        return (out_grad * exp(node.inputs[0]),)


def exp(a):
    return Exp()(a)



class ReLU(TensorOp):
    def compute(self, a: NDArray):
        out = np.copy(a)
        out[a < 0] = 0
        return out

    def gradient(self, out_grad: "Tensor", node: "Tensor"):
        out = node.realize_cached_data().copy()
        out[out > 0] = 1
        return (out_grad * Tensor(out),)


def relu(a):
    return ReLU()(a)

    

def sigmoid(a):
    return (1 + exp(-a)) ** (-1)



def softmax(a, dim=None):
    a = a.exp()

    new_shape = list(a.shape)
    new_shape[dim] = 1
    a_sum = a.sum(dim=(dim,)).reshape(tuple(new_shape)).broadcast_to(a.shape)

    return a / a_sum


class Value:
    op: Optional[Op] # 节点对应的计算操作， Op是自定义的计算操作类
    inputs: List["Value"]
    cached_data: NDArray
    requires_grad: bool
    def _init(self, op: Optional[Op], inputs: List["Value"], *, num_outputs: int = 1,
    cached_data: NDArray = None, requires_grad: Optional[bool] = None):
        global TENSOR_COUNTER
        TENSOR_COUNTER += 1
        if requires_grad is None:
            requires_grad = any(x.requires_grad for x in inputs)

        self.op = op
        self.inputs = inputs
        self.num_outputs = num_outputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad

    def realize_cached_data(self): 
        # 进行计算得到节点对应的变量，存储在cached_data里
        if self.is_leaf() or self.cached_data is not None:
            return self.cached_data
        self.cached_data = self.op.compute(*[x.realize_cached_data() for x in self.inputs])
        return self.cached_data
    
    def is_leaf(self):
        return self.op is None
    
    def __del__(self):
        global TENSOR_COUNTER
        TENSOR_COUNTER -= 1

    @classmethod
    def make_const(cls, data, *, requires_grad=False):
        # 建立一个用data生成的独立节点
        value = cls.__new__(cls)
        value._init(None, [], cached_data=data, requires_grad=requires_grad)
        return value

    @classmethod
    def make_from_op(cls, op: Op, inputs: List["Value"]):
        # 根据op生成节点
        value = cls.__new__(cls)
        value._init(op, inputs)

        if not value.requires_grad:
            return value.detach()
        
        value.realize_cached_data()
        return value



class Tensor(Value):
    grad: "Tensor"
    def __init__(self, array, *, dtype=None, requires_grad=True, **kwargs):
        if isinstance(array, Tensor):
            if dtype is None:
                dtype = array.dtype
            if dtype == array.dtype:
                cached_data = array.realize_cached_data()
            else:
                cached_data = np.array(array.realize_cached_data(), dtype=dtype)
        else:
            cached_data = np.array(array, dtype=dtype)

        self._init(None, [], cached_data=cached_data, requires_grad=requires_grad)

    @staticmethod
    def from_numpy(numpy_array, dtype):
        tensor = Tensor.__new__(Tensor)
        cached_data = np.array(numpy_array, dtype=dtype)
        tensor._init(None, [], cached_data = cached_data)
        return tensor
    
    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"]):
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
        tensor.realize_cached_data()
        return tensor
    
    @staticmethod
    def make_const(data, requires_grad=False):
        tensor = Tensor.__new__(Tensor)
        if isinstance(data, Tensor):
            data = data.realize_cached_data()
        tensor._init(None, [], cached_data = data, requires_grad = requires_grad)
        return tensor
    
    def detach(self):
        # 创建一个新的张量，但不接入计算图
        return Tensor.make_const(self.realize_cached_data())
    
    @property
    def data(self): 
        #对cached_data进行封装
        return self.detach()
    
    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor)
        self.cached_data = value.realize_cached_data()

    @property
    def shape(self):
        return self.realize_cached_data().shape

    @property
    def dtype(self):
        return self.realize_cached_data().dtype
    
    def __repr__(self):
        return "Tensor(" + str(self.realize_cached_data()) + ")"
    
    def backward(self, out_grad=None):
        # 最后一个节点时，out_grad为1
        if out_grad is None:
            out_grad = Tensor(np.ones(self.shape))
        compute_gradient_of_variables(self, out_grad)

    def numpy(self):
        return self.realize_cached_data()

    def __add__(self, other):
        if isinstance(other, Tensor):
            return EWiseAdd()(self, other)
        else:
            return AddScalar(other)(self)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return EWiseAdd()(self, Negate()(other))
        else:
            return AddScalar(-other)(self)
        
    def __rsub__(self, other):
        return AddScalar(other)(-self)
        
    def __mul__(self, other):
        if isinstance(other, Tensor):
            return EWiseMul()(self, other)
        else:
            return MulScalar(other)(self)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            raise NotImplementedError()
        else:
            return PowerScalar(other)(self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return EWiseDiv()(self, other)
        else:
            return DivScalar(other)(self)
    
    def transpose(self, dim=None):
        return Transpose(dim)(self)
    
    def reshape(self, shape):
        return Reshape(shape)(self)

    def broadcast_to(self, shape):
        return BroadcastTo(shape)(self)
    
    def sum(self, dim=None):
        return Summation(dim)(self)

    def __matmul__(self, other):
        return MatMul()(self, other)
    
    def matmul(self, other):
        return MatMul()(self, other)
    
    def __neg__(self):
        return Negate()(self)
    
    def log(self):
        return Log()(self)
    
    def exp(self):
        return Exp()(self)
    
    def relu(self):
        return ReLU()(self)
    
    def sigmoid(self):
        return sigmoid(self)
    
    __radd__ = __add__
    __rmul__ = __mul__
    __rmatmul__ = __matmul__
    


def find_topo_sort(node_list: List[Value]) -> List[Value]:
    topo_order = []
    visited = set()

    def dfs(node):
        nonlocal topo_order, visited

        visited.add(node)
        for input_node in node.inputs:
            if input_node not in visited:
                dfs(input_node)
        topo_order.append(node)

    for node in node_list:
        if node not in visited:
            dfs(node)

    return topo_order


def compute_gradient_of_variables(output_tensor, out_grad):
    node_to_output_grads_list: Dict[Tensor, List[Tensor]] = defaultdict(list) # dict结构，用于存储partial adjoint
    node_to_output_grads_list[output_tensor] = [out_grad]
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor]))) # 请自行实现拓扑排序函数

    for node in reverse_topo_order:
        node.grad = reduce(add, node_to_output_grads_list[node]) # 求node的partial adjoint之和，存入属性grad
        if node.is_leaf():
            continue
        for i, grad in enumerate(node.op.gradient(node.grad, node)): # 计算node.inputs的partial adjoint
            j = node.inputs[i]
            node_to_output_grads_list[j].append(grad) # 将计算出的partial adjoint存入dict