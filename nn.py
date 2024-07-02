from abc import ABC, abstractmethod
from typing import List
import numpy as np

from autograd import Tensor
import init
import autograd as ag


class Parameter(Tensor): 
    # 声明一个类专门表示网络参数
    pass


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
    


class Module(ABC): 
    def __init__(self):
        self.training = True

    def parameters(self) -> List["Tensor"]:
        return _unpack_params(self.__dict__) 
    
    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__) 
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    @abstractmethod
    def forward(self): 
        pass

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True



class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(init.kaiming_normal(in_features, out_features, dtype=dtype)) #请自行实现初始化算法 
        if bias:
            self.bias = Parameter(Tensor(np.zeros(self.out_features)))
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor: 
        X_out = X @ self.weight
        if self.bias is not None:
            return X_out + self.bias.broadcast_to(X_out.shape)
        return X_out
    


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor: 
        return X.reshape((X.shape[0], np.prod(X.shape[1:])))
    


class ReLU(Module):
    def forward(self, X: Tensor) -> Tensor:
        return ag.relu(X)
    


class Sigmoid(Module):
    def forward(self, X: Tensor) -> Tensor:
        return ag.sigmoid(X)



class Softmax(Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, X: Tensor) -> Tensor:
        return ag.softmax(X, self.dim)
    


class CrossEntropyLoss(Module):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert len(input.shape) == 2
        assert len(target.shape) == 1
        assert input.shape[0] == target.shape[0]

        softmax_input = ag.softmax(input, dim=1)
        y = np.zeros(input.shape)
        for i, c in enumerate(target.numpy()):
            y[i, c] = 1
        y_tensor = Tensor(y)

        return ag.summation(-y_tensor * softmax_input.log()) / input.shape[0]
    


class BinaryCrossEntropyLoss(Module):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert input.shape == target.shape
        assert len(input.shape) == 2

        return -(target * input.log() + (1-target) * (1-input).log()).sum() / np.prod(input.shape)

        

class MSELoss(Module):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert input.shape == target.shape
        return ag.summation((input - target) ** 2) / np.prod(input.shape)



class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules
        
    def forward(self, x: Tensor) -> Tensor: 
        for module in self.modules:
            x = module(x) 
        return x
    


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fn(x)