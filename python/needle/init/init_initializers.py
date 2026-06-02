import math
from .init_basic import *
from typing import Any


def xavier_uniform(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    a=gain*((6/(fan_in+fan_out))**0.5)
    return rand(*(fan_in,fan_out),low=-a,high=a)
    ### END YOUR SOLUTION


def xavier_normal(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    std=gain*((2/(fan_in+fan_out))**0.5)
    return randn(*(fan_in,fan_out),mean=0,std=std)
    ### END YOUR SOLUTION

def kaiming_uniform(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    bound=(2**0.5)*((3/fan_in)**0.5)
    return rand(*(fan_in,fan_out),low=-bound,high=bound)
    ### END YOUR SOLUTION



def kaiming_normal(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    std=(2**0.5)/(fan_in**0.5)
    return randn(*(fan_in,fan_out),mean=0,std=std)
    ### END YOUR SOLUTION