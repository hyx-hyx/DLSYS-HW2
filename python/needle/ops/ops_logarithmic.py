from typing import Optional, Any, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        maxz=array_api.max(Z,axis=1,keepdims=True)
        log_sum_exp_z=array_api.log(array_api.sum(array_api.exp(Z-maxz),axis=1,keepdims=True))
        return Z-log_sum_exp_z-maxz
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        sum_out_grad=reshape(summation(out_grad,axes=1),(out_grad.shape[0],1))
        softmax=exp(logsoftmax(node.inputs[0]))
        return (out_grad-softmax*sum_out_grad,)
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        maxz=array_api.max(Z,self.axes,keepdims=True)
        sum_exp=array_api.sum(array_api.exp(Z-maxz),axis=self.axes,keepdims=True)
        return array_api.squeeze(array_api.log(sum_exp)+maxz,self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        
        if self.axes is not None:
            keep_dims_shape=list(Z.shape)
            for axis in self.axes:
                keep_dims_shape[axis]=1
        else:
            keep_dims_shape=[1]*len(Z.shape)
            
        # 计算 softmax
        softmax=exp(Z-reshape(logsumexp(Z,self.axes),tuple(keep_dims_shape)))
        out_grad=reshape(out_grad,keep_dims_shape)
        return (out_grad*softmax,)
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)