# DLSYS HW2 SoftmaxLoss 相关修复说明

## 问题背景

本次修复涉及两个测试用例的调试：
- `test_nn_softmax_loss_backward_1`：SoftmaxLoss 反向传播正确性测试（5×10）
- `test_nn_softmax_loss_backward_2`：SoftmaxLoss 反向传播正确性测试（3×11）

---

## 修改文件及内容

### 1. `python/needle/nn/nn_basic.py`（Line 9）

```python
# 错误 ❌
from python import needle

# 正确 ✅
import needle
```

### 2. `python/needle/ops/ops_logarithmic.py`（4处）

| 修改位置 | 问题 | 解决 |
|---------|------|------|
| Line 3 | `from python import needle` | 删除（不再使用） |
| `LogSoftmax.compute` (Line 13-17) | 在 compute 中创建内部 Tensor 计算图节点 | 改为纯 numpy 计算 |
| `LogSoftmax.gradient` (Line 21-28) | 混用 numpy 和 Tensor ops | 全部使用 Tensor ops |
| `LogSumExp.gradient` (Line 38-52) | 全部 numpy 计算后包 `Tensor()` | 全部使用 Tensor ops |

---

## 根因分析

### 核心问题：两个不同的 `Tensor` 类

修改前的关键代码：

```python
# nn_basic.py（Line 4, Line 9）
from needle.autograd import Tensor  # 导入 needle.autograd.Tensor（类 A）
from python import needle            # needle 被覆盖为 python.needle（类 B）
```

```python
# ops_logarithmic.py（Line 3, Line 5）
from python import needle            # needle 被覆盖为 python.needle（类 B）
from ..autograd import Tensor        # 导入 needle.autograd.Tensor（类 A）
```

由于 `python/` 目录下没有 `__init__.py`，Python 3 将其识别为 **namespace package**。而 `./python` 被添加在 `sys.path` 上，`needle` 包可以直接作为 `needle` 访问。

这导致 Python 的模块系统将 `needle` 和 `python.needle` 视为**两个不同的模块路径**，分别加载了 `needle/autograd.py`，产生了**两个不同的 `Tensor` 类**：

- `needle.autograd.Tensor` → 类 A（由 `from ..autograd import Tensor` 和 `from needle.autograd import Tensor` 导入）
- `python.needle.autograd.Tensor` → 类 B（由 `from python import needle` 导入链产生）

### 错误传播链

```
SoftmaxLoss.forward 中：
  needle.summation(...)          ← needle 是 python.needle（类 B）
  needle.logsumexp(...)          ← needle 是 python.needle（类 B）

EWiseMul.gradient 中：
  return out_grad * rhs         ← __mul__ 中的 isinstance(rhs, Tensor) 检查
                                  rhs 是类 B 的 Tensor，但 Tensor 是类 A → 返回 False
```

```python
# Tensor.__mul__（autograd.py）
def __mul__(self, other):
    if isinstance(other, Tensor):    # 类 A 的 isinstance 检查类 B → False！
        return EWiseMul()(self, other)
    else:
        return MulScalar(other)(self) # 走到这里！Tensor 被当作"标量"
```

### dtype=object 的诞生

```python
# MulScalar.compute
def compute(self, a):
    return a * self.scalar
    # a 是 float32 numpy 数组
    # self.scalar 是 Tensor 对象
    # numpy 对未知对象逐元素广播 → dtype=object 的数组
    # 每个元素是一个 needle.Tensor！
```

### 最终报错

object-dtype 数组一路传播到任意后续 Op（如 `Exp.compute`），调用 `np.exp()` 时报错：

```
TypeError: loop of ufunc does not support argument 0 of type Tensor
which has no callable exp method
```

### 可视化流程图

```
from python import needle
        │
        ▼
python.needle.Tensor (类 B)  ≠  needle.autograd.Tensor (类 A)
        │
        ▼
isinstance(tensor_B, Tensor_A) → False
        │
        ▼
Tensor_A.__mul__(tensor_B) → MulScalar(tensor_B)(tensor_A)
        │
        ▼
numpy_array * Tensor_B → dtype=object 的数组
        │
        ▼
后续所有 Op 的 compute 收到 object 数组 → TypeError
```

---

## 各修改详解

### 修改 1：删除/修正 `from python import needle`

**文件**：`nn_basic.py` Line 9, `ops_logarithmic.py` Line 3

**原因**：`from python import needle` 是导致两个 Tensor 类共存的根本原因。修复方式：
- `nn_basic.py`：改为 `import needle`，确保使用与 `from needle.autograd import Tensor` 一致的模块路径
- `ops_logarithmic.py`：直接删除，因为文件中不再使用 `needle` 引用

---

### 修改 2：`LogSoftmax.compute` — 纯 numpy 实现

**文件**：`ops_logarithmic.py`

```python
# 错误 ❌：在 compute 内部创建 Tensor 图节点
return (Tensor(Z)-reshape(logsumexp(Tensor(Z),axes=(1,)),
       (Z.shape[0],1))).realize_cached_data()

# 正确 ✅：compute 应该只做纯 numpy 计算
max_z = array_api.max(Z, axis=1, keepdims=True)
log_sum_exp = array_api.log(array_api.sum(array_api.exp(Z - max_z),
                            axis=1, keepdims=True)) + max_z
return Z - log_sum_exp
```

**原因**：`compute` 方法在 `realize_cached_data()` 中被调用，它的职责是用 numpy 计算该 Op 的前向值。在 compute 内部通过 `Tensor(Z)` 创建新叶子节点、调用 `logsumexp`/`reshape` 等 Tensor ops，会在*计算图中*创建了额外的内部节点。这些节点虽然不影响本次测试（因为 `SoftmaxLoss` 不使用 `LogSoftmax`），但逻辑上是不正确的，可能导致其他场景的 bug。

**LogSoftmax 的数学公式**：

$$\text{logsoftmax}(x_i) = x_i - \log\left(\sum_j e^{x_j}\right)$$

通过减去最大值保证数值稳定性：

$$\text{logsoftmax}(x_i) = x_i - \log\left(\sum_j e^{x_j - \max(x)}\right) - \max(x)$$

---

### 修改 3：`LogSoftmax.gradient` — 纯 Tensor ops

**文件**：`ops_logarithmic.py`

```python
# 错误 ❌：混用 numpy 和 Tensor ops
softmax = needle.exp(logsoftmax(node.inputs[0])).realize_cached_data()
return (out_grad-softmax*array_api.sum(out_grad.realize_cached_data(),
        axis=1, keepdims=True),)

# 正确 ✅：全部使用 Tensor ops
Z = node.inputs[0]
softmax = exp(logsoftmax(Z))
sum_grad = summation(out_grad, axes=(1,))
sum_grad_reshaped = reshape(sum_grad, (Z.shape[0], 1))
return (out_grad - sum_grad_reshaped * softmax,)
```

**原因**：梯度函数接收 Tensor 参数，应返回 Tensor 表达式来构建*梯度计算图*。原代码：
1. `.realize_cached_data()` 强制求值 → 切断了梯度链
2. `softmax * array_api.sum(...)` 是 numpy 乘法 → 得到 numpy 数组
3. `out_grad - numpy_array` → `Tensor.__sub__(numpy_array)` → `AddScalar(-numpy_array)(out_grad)` — 将整个 numpy 数组当作 AddScalar 的"标量"，创建了非标准的计算图节点

**LogSoftmax 的梯度公式**（雅可比矩阵的向量-雅可比积）：

$$\frac{\partial \text{logsoftmax}(x)_i}{\partial x_j} = \delta_{i,j} - \text{softmax}(x)_j$$

$$\text{grad}_x = \text{out\_grad} - \text{softmax}(x) \cdot \sum_j \text{out\_grad}_j$$

---

### 修改 4：`LogSumExp.gradient` — 纯 Tensor ops

**文件**：`ops_logarithmic.py`

```python
# 错误 ❌：全部 numpy 计算，最后包 Tensor()
input_data = input.realize_cached_data()
max_input = array_api.max(input_data, self.axes, keepdims=True)
...
grad_data = out_grad_data * softmax
return (Tensor(grad_data),)

# 正确 ✅：全部使用 Tensor ops
Z = node.inputs[0]
keep_dims_shape = ...
softmax = exp(Z - reshape(logsumexp(Z, self.axes), tuple(keep_dims_shape)))
out_grad = reshape(out_grad, keep_dims_shape)
return (out_grad * softmax,)
```

**原因**：
1. `Tensor(grad_data)` 创建了一个**新的叶子 Tensor**，默认 `requires_grad=True`，当它被多个梯度贡献源通过 `sum_node_list` 求和时，可能与异源 Tensor 产生不兼容
2. 如果存在前述的 Tensor 类冲突，这个叶子 Tensor 的类型与其他 Tensor 不一致，导致算术操作进入错误分支

**LogSumExp 的梯度公式**：

$$\frac{\partial \text{logsumexp}(x)}{\partial x_i} = \frac{e^{x_i}}{\sum_j e^{x_j}} = \text{softmax}(x)_i$$

$$\text{grad}_x = \text{broadcast(out\_grad)} \cdot \text{softmax}(x)$$

**数值稳定计算**（全部用 Tensor ops）：

$$\text{softmax}(x) = \exp\left(x - \text{reshape}\left(\text{logsumexp}(x, \text{axes}), \text{keepdims}\right)\right)$$

---

## 核心教训

1. **Python 模块导入一致性**：永远不要混用 `from python.xxx import ...` 和 `from needle.xxx import ...` 来导入同一个包。namespace package 会导致模块被重复加载，产生重复的类定义。

2. **梯度函数应使用 Tensor ops 构建计算图**：不应该在 gradient 中调用 `.realize_cached_data()` 急于求值，也不应该用 numpy 数组手动计算然后包 `Tensor()`。梯度函数应该像前向传播一样，使用 Tensor 操作来定义梯度计算。

3. **compute 函数应使用纯 numpy**：不要在 compute 中创建 Tensor 对象或调用 Tensor ops，这会污染计算图。
