from abc import ABC, abstractmethod
from collections import defaultdict
import math


class Optimizer(ABC):
    def __init__(self, params):
        self.params = params

    @abstractmethod
    def step(self):
        pass

    def reset_grad(self):
        for p in self.params:
            p.grad = None



class SGD(Optimizer):
    def __init__(self, params, lr=0.001, momentum=0., weight_decay=0.):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = defaultdict(float)
        self.weight_decay = weight_decay

    def step(self):
        for w in self.params:
            if self.weight_decay > 0:
                grad = w.grad.data + self.weight_decay * w.data
            else:
                grad = w.grad.data
            self.u[w] = self.momentum * self.u[w] + (1 - self.momentum) * grad
            w.data = w.data - self.lr * self.u[w]



class Adam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0,999), eps=1e-8, weight_decay=0.):
        super().__init__(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = defaultdict(float)
        self.v = defaultdict(float)

    def step(self):
        self.t += 1
        for w in self.params:
            if self.weight_decay > 0:
                grad = w.grad.data + self.weight_decay * w.data
            else:
                grad = w.grad.data
            self.m[w] = self.betas[0] * self.m[w] + (1 - self.betas[0]) * grad
            self.v[w] = self.betas[1] * self.v[w] + (1 - self.betas[1]) * (grad ** 2)
            unbiased_m = self.m[w] / (1 - self.betas[0] ** self.t)
            unbiased_v = self.v[w] / (1 - self.betas[1] ** self.t)
            w.data = w.data - self.lr * unbiased_m / (unbiased_v**0.5 + self.eps)



class Scheduler(ABC):
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.step_count = 0

    @abstractmethod
    def step(self):
        pass



class StepDecay(Scheduler):
    def __init__(self, optimizer, step_size, gamma=0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self):
        self.step_count += 1
        if self.step_count % self.step_size == 0:
            self.optimizer.lr *= self.gamma



class LinearWarmUp(Scheduler):
    def __init__(self, optimizer, warm_up_steps, start_lr, end_lr):
        super().__init__(optimizer)
        self.warm_up_steps = warm_up_steps
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warm_up_steps:
            lr = self.start_lr + (self.end_lr - self.start_lr) * self.step_count / self.warm_up_steps
        else:
            lr = self.end_lr
        self.optimizer.lr = lr



class CosineDecayWithWarmRestarts(Scheduler):
    def __init__(self, optimizer, T_max, eta_min=0., warm_up_steps=0, T_mult=1.):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min
        self.warm_up_steps = warm_up_steps
        self.T_mult = T_mult
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warm_up_steps:
            lr = self.eta_min + (self.T_max - self.eta_min) * self.step_count / self.warm_up_steps
        else:
            T_cur = self.T_max * (self.T_mult ** (self.step_count // self.T_mult))
            lr = self.eta_min + (self.T_max - self.eta_min) * (1 + math.cos(math.pi * (self.step_count % self.T_mult) / T_cur)) / 2
        self.optimizer.lr = lr