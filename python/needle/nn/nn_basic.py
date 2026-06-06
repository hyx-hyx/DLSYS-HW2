"""The module.
"""
from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
import needle

from python.needle.init.init_initializers import kaiming_uniform
from python.needle.init.init_basic import randb

class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
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


def _child_modules(value: object) -> list["Module"]:
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
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight=Parameter(kaiming_uniform(in_features,out_features))
        if bias:
            self.bias=Parameter(kaiming_uniform(fan_in=out_features,fan_out=1).transpose())
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if 'bias' in self.__dict__:
            return X.matmul(self.weight)+self.bias.broadcast_to((X.shape[0],self.out_features))
        else:
            return X.matmul(self.weight)
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        dim_2=1
        for axis in X.shape[1:]:
            dim_2*=axis
        return needle.reshape(X,(X.shape[0],dim_2))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return needle.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out=x
        for module in self.modules:
            out=module(out)
        return out
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        z_y=needle.summation(logits*init.one_hot(n=logits.shape[1],i=y))
        return (needle.summation(needle.logsumexp(logits,axes=(1,)))-z_y)/logits.shape[0]
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight=Parameter(array=init.ones(dim))
        self.bias=Parameter(array=init.zeros(dim))
        self.running_mean=Parameter(array=init.zeros(dim))
        self.running_var=Parameter(array=init.ones(dim))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            dim=self.dim
            x_s1=x.shape[1]
            
            # 计算mean
            mean=needle.summation(x,axes=(0,))/x.shape[0]
            reshape_mean=needle.reshape(mean,(1,x_s1))
            
            # 计算var
            x_minus_mean=x-needle.broadcast_to(reshape_mean,x.shape)
            var=needle.summation(x_minus_mean**2,axes=(0,))/x.shape[0]
            reshape_var=needle.reshape(var,(1,x_s1))
            var_broad_cast=needle.broadcast_to(reshape_var,x.shape)
            
            # 更新running_mean
            self.running_mean=self.running_mean*(1-self.momentum)+mean*self.momentum
            
            # 更新running_var
            self.running_var=self.running_var*(1-self.momentum)+var*self.momentum
        else:
            reshape_mean=needle.reshape(self.running_mean,(1,x_s1))
            x_minus_mean=x-needle.broadcast_to(reshape_mean,x.shape)
            
            reshape_var=needle.reshape(self.running_var,(1,x_s1))
            var_broad_cast=needle.broadcast_to(reshape_var,x.shape)
            
        norm=x_minus_mean/(var_broad_cast+self.eps)**0.5
        return needle.broadcast_to(self.weight,x.shape)*norm+needle.broadcast_to(needle.reshape(self.bias,(1,dim)),x.shape)
        ### END YOUR SOLUTION



class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight=Parameter(array=init.ones(dim))
        self.bias=Parameter(array=init.zeros(dim))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        dim=self.dim
        mean=needle.reshape(needle.summation(x,axes=(1,))/dim,(x.shape[0],1))
        x_minus_mean=x-needle.broadcast_to(mean,x.shape)
        var=needle.reshape(needle.summation(x_minus_mean**2,axes=(1,))/dim,(x.shape[0],1))
        var_broad_cast=needle.broadcast_to(var,x.shape)
        norm=x_minus_mean/(var_broad_cast+self.eps)**0.5
        return needle.broadcast_to(self.weight,x.shape)*norm+needle.broadcast_to(needle.reshape(self.bias,(1,dim)),x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            return x*randb(*x.shape,p=self.p,dtype=float)/(1-self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x)+x
        ### END YOUR SOLUTION
