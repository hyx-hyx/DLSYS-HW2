from typing import Optional, Any, Union

from python import needle
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return (Tensor(Z)-logsumexp(Tensor(Z),axes=(1,))).realize_cached_data()
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        maxz=array_api.max(Z,self.axes,keepdims=True)
        sum_exp=array_api.sum(array_api.exp(Z-maxz),axis=self.axes)
        if self.axes is None:
            maxz=maxz.reshape(())
        else:
            # 去掉归约轴
            maxz = array_api.squeeze(maxz, axis=self.axes)
        return array_api.log(sum_exp)+maxz
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        input=node.inputs[0]
        sum_exp=needle.summation(needle.exp(input),axes=self.axes)
        sum_input=needle.reshape(sum_exp,(input.shape[0],1))
        out_grad_reshape=needle.reshape(out_grad,(input.shape[0],1))
        return (multiply(out_grad_reshape,(needle.exp(input)/sum_input)),)
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)