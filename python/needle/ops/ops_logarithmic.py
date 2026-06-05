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
        return (Tensor(Z)-reshape(logsumexp(Tensor(Z),axes=1),(Z.shape[0],1))).realize_cached_data()
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
        sum_exp=array_api.sum(array_api.exp(Z-maxz),axis=self.axes,keepdims=True)
        return array_api.squeeze(array_api.log(sum_exp)+maxz,self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        input = node.inputs[0]
        
        # 计算 softmax
        input_data = input.realize_cached_data()
        max_input = array_api.max(input_data, self.axes, keepdims=True)
        stable_input = input_data - max_input
        exp_stable = array_api.exp(stable_input)
        sum_exp = array_api.sum(exp_stable, axis=self.axes, keepdims=True)
        softmax = exp_stable / sum_exp
        
        # 处理 out_grad 维度
        out_grad_data = out_grad.realize_cached_data()
        
        # 如果维度不匹配，调整 out_grad
        if out_grad_data.shape != softmax.shape:
            # 确定在哪些轴插入维度
            axes_to_expand = self.axes if self.axes is not None else tuple(range(softmax.ndim))
            
            # 插入维度
            for axis in sorted(axes_to_expand):
                out_grad_data = array_api.expand_dims(out_grad_data, axis=axis)
        
        # 计算梯度
        grad_data = out_grad_data * softmax
        
        return (Tensor(grad_data),)
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)