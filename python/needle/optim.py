"""Optimization module"""
import needle as ndl
import needle.init as init
import numpy as np

from .autograd import Tensor


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
        # BEGIN YOUR SOLUTION
        for para in self.params:
            shape = para.shape
            grad = para.grad + self.weight_decay * para
            if para not in self.u.keys():
                self.u[para] = init.zeros(*shape)
            momentum = Tensor([self.momentum])
            old_u = self.u[para].detach()
            self.u[para] = ndl.broadcast_to(momentum * old_u, shape) + (init.ones(*momentum.shape) - momentum) * grad
            para.data = para.data - self.lr * Tensor(self.u[para], dtype='float32')
        # END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        Note: This does not need to be implemented for HW2 and can be skipped.
        """
        # BEGIN YOUR SOLUTION
        raise NotImplementedError()
        # END YOUR SOLUTION


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
        # BEGIN YOUR SOLUTION
        self.t += 1
        t = self.t
        for para in self.params:
            shape = para.shape
            grad = para.grad + ndl.broadcast_to(Tensor([self.weight_decay], dtype='float32'), shape) * para

            beta1 = ndl.broadcast_to(Tensor([self.beta1]), shape)
            beta2 = ndl.broadcast_to(Tensor([self.beta2]), shape)

            if para not in self.m.keys():
                self.m[para] = init.zeros(*shape)
            if para not in self.v.keys():
                self.v[para] = init.zeros(*shape)

            old_u = self.m[para].detach()
            old_v = self.v[para].detach()

            self.m[para] = (beta1 * old_u + (init.ones(*shape) - beta1) * grad)
            m = self.m[para] / (init.ones(*shape) - beta1**t)
            self.v[para] = (beta2 * old_v + (init.ones(*shape) - beta2) * (grad**2))
            v = self.v[para] / (init.ones(*shape) - beta2**t)

            para.data = para.data - Tensor(self.lr * m / (v**0.5 + self.eps), dtype='float32')
        # END YOUR SOLUTION
