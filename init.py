import math
import numpy as np
from autograd import Tensor


def rand(*shape, low=0.0, high=1.0, dtype="float32", requires_grad=False):
    array = np.random.rand(*shape) * (high - low) + low
    return Tensor(array, dtype=dtype, requires_grad=requires_grad)


def randn(*shape, mean=0.0, std=1.0, dtype="float32", requires_grad=False):
    array = np.random.randn(*shape) * std + mean
    return Tensor(array, dtype=dtype, requires_grad=requires_grad)


def constant(*shape, c=1.0, dtype="float32", requires_grad=False):
    array = np.ones(*shape, dtype=dtype) * c
    return Tensor(array, dtype=dtype, requires_grad=requires_grad)


def zeros(*shape, dtype="float32", requires_grad=False):
    return constant(*shape, c=0.0, dtype=dtype, requires_grad=requires_grad)


def ones(*shape, dtype="float32", requires_grad=False):
    return constant(*shape, c=1.0, dtype=dtype, requires_grad=requires_grad)


def one_hot(num, idx, dtype="float32", requires_grad=False):
    array = np.eye(num)[idx]
    return Tensor(array, dtype=dtype, requires_grad=requires_grad)


def kaiming_uniform(fan_in, fan_out, mode='fan_in', **kwargs):
    if mode == 'fan_in':
        bound = math.sqrt(6 / fan_in)
    else:
        bound = math.sqrt(6 / fan_out)
    return rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)


def kaiming_normal(fan_in, fan_out, mode='fan_in', **kwargs):
    if mode == 'fan_in':
        std = math.sqrt(2 / fan_in)
    else:
        std = math.sqrt(2 / fan_out)
    return randn(fan_in, fan_out, mean=0, std=std, **kwargs)


def xavier_uniform(fan_in, fan_out, gain, **kwargs):
    a = gain * math.sqrt(6 / (fan_in + fan_out))
    return rand(fan_in, fan_out, low=-a, high=a, **kwargs)


def xavier_normal(fan_in, fan_out, gain, **kwargs):
    std = gain * math.sqrt(2 / (fan_in + fan_out))
    return randn(fan_in, fan_out, mean=0, std=std, **kwargs)