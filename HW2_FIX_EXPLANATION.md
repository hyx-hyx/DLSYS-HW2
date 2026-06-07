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


---

# DLSYS HW2 Adam 优化器偏差校正错误

## 问题背景

涉及的测试用例：
- `test_optim_adam_1`：Adam 优化器正确性测试，1 epoch 训练后期望最终 loss 为 3.703999

**实际输出**: `6.22266` | **期望输出**: `3.703999` — 实际 loss 远高于期望，说明优化器收敛显著偏慢。

---

## 问题位置

**文件**：`python/needle/optim.py`，`Adam.step()` 方法（第 71-94 行）

---

## 核心问题：偏差校正被内嵌入存储值，污染了下一轮的递推

### 错误代码（Line 90-91）

```python
self.m[para]=(beta1*old_u+(init.ones(*shape)-beta1)*grad)/(init.ones(*shape)-beta1**t)
self.v[para]=(beta2*old_v+(init.ones(*shape)-beta2)*(grad**2))/(init.ones(*shape)-beta2**t)
```

**这段代码将偏差校正整合进了 `self.m[para]` 和 `self.v[para]` 的存储值中**——也就是说，存储的是偏差校正后的估计量（`m_hat_t`、`v_hat_t`），而非原始（有偏）估计量（`m_t`、`v_t`）。

### 错误传播链

```
Step t=1:
  old_u = 0（初始化为零）
  m₁ = (beta1·0 + (1-beta1)·g₁) / (1-beta1¹)     ← 存储 m_hat₁ ✓（t=1 时恰好正确）

Step t=2:
  old_u = self.m[para].detach()                    ← 读取的是 m_hat₁，而非 m₁！
  m₂ = (beta1·m_hat₁ + (1-beta1)·g₂) / (1-beta1²)  ← ❌ 用 m_hat₁ 替代 m₁ 做递推！

Step t=3:
  old_u → m_hat₂（已被污染的）
  m₃ = (beta1·m_hat₂ + (1-beta1)·g₃) / (1-beta1³)  ← ❌ 污染继续累积
```

### 正确的 Adam 算法（参考 PyTorch 实现）

```
# 递推在原始（有偏）空间中进行
m_t     = beta1 * m_{t-1}     + (1-beta1) * g_t        # 存储原始值
v_t     = beta2 * v_{t-1}     + (1-beta2) * g_t²       # 存储原始值

# 偏差校正仅在参数更新时使用
m_hat_t = m_t / (1 - beta1^t)
v_hat_t = v_t / (1 - beta2^t)

param   = param - lr * m_hat_t / (sqrt(v_hat_t) + eps)
```

**关键区别**：存储的始终是原始（有偏）的 `m` / `v`，偏差校正仅在参数更新的分母中应用一次，不污染存储值。

---

## 数值影响分析

由于 `m_hat_{t-1} = m_{t-1} / (1 - beta1^{t-1})`，在下一轮的递推中：

```
错误代码实际计算：
  m̂_new = (beta1 · [m_{t-1} / (1-beta1^{t-1})] + (1-beta1) · g_t) / (1-beta1^t)

正确公式：
  m̂_new = (beta1 · m_{t-1}                       + (1-beta1) · g_t) / (1-beta1^t)
```

以 `beta1=0.9, t=2` 为例：
- `1 - beta1^1 = 0.1`，故 `m_hat₁ = m₁ / 0.1 = 10 × m₁`
- `m₁ = (1 - 0.9) × g₁ = 0.1 × g₁`
- **错误计算**：`(0.9 × 10m₁ + 0.1g₂) / 0.19 = (9m₁ + 0.1g₂) / 0.19`（旧梯度权重 = 9）
- **正确计算**：`(0.9 × m₁ + 0.1g₂) / 0.19`（旧梯度权重 = 0.9）

**代码对旧的梯度估计量赋予了 `1/(1-beta1^{t-1})` 倍的错误权重。** 当 `beta1=0.9, t=2` 时，这个倍数就是 10 倍。随着 t 增大，这个偏差系数逐渐趋近于 1，但在早期步骤中，它使得优化器严重偏向初始梯度方向，对新梯度反应迟钝，导致收敛速度大幅下降。1 epoch 后 loss 为 6.22（vs 正确的 3.70），正是这种"惯性过大"效应的体现。

### 可视化

```
             旧梯度的有效权重 (beta1=0.9)
             
t=2:  正确=0.9     错误=9.0    (10×)
t=3:  正确=0.9     错误≈4.74   (5.3×)  
t=10: 正确=0.9     错误≈1.38   (1.5×)
...收敛到 0.9
```

---

## 修复方向

将偏差校正从存储值中分离出来，仅在第 93 行的参数更新步骤中应用：

1. **Line 90-91**：存储原始（有偏）的 `m` 和 `v`，不做偏差校正除法
2. **Line 93**：在参数更新时，分别对 `self.m[para]` 除以 `1 - beta1^t`、对 `self.v[para]` 除以 `1 - beta2^t` 以得到偏差校正后的估计量


---
---
# DLSYS HW2 SGD Momentum 内存泄漏 & 动量状态丢失

## 问题背景

涉及的测试用例：
- `test_optim_sgd_momentum_1`：SGD momentum 优化器正确性测试，期望 loss 3.311805
- `test_optim_sgd_z_memory_check_1`：SGD 系列测试后的全局 Tensor 计数检查，期望 ~387（连续运行全部 SGD 测试时触发）

**现象**：单独运行任一个测试均通过，但连续运行所有 `test_optim_sgd_*` 测试时 memory_check 失败（实际 tensor 计数 4720，远超期望）。

---

## 根因一：字典键用 `grad` 导致动量状态每次被清零

### 问题位置

`python/needle/optim.py`，`SGD.step()` 方法

### 错误代码

```python
for para in self.params:
    grad = para.grad + self.weight_decay * para
    if grad not in self.u.keys():   # ❌ 用 grad 做字典键
        self.u[grad]=0             # ❌ 用 grad 做字典键
    ...
    self.u[grad] = ...              # ❌ 每次 step 产生新 key
```

### 错误传播链

```
每次 step() 调用：
  backward() 产生全新的 grad Tensor 对象（不同 Python id）
     ↓
  grad not in self.u.keys() → 永远为 True（新对象永远不在旧字典中）
     ↓
  self.u[grad] = 0 → 动量缓冲区被重置！
     ↓
  momentum * 0 + (1-momentum) * grad → 等效 lr = lr × (1-momentum)
     ↓
  动量从未真正累积，模型严重欠拟合
```

### 影响

- **Vanilla SGD（momentum=0）**：恰巧正常工作，因为 `momentum × 0 = 0`，`u` 始终等于 `grad`，重置与否无影响
- **Momentum SGD（momentum=0.9）**：`u = 0.9 × 0 + 0.1 × grad`，等效学习率缩水 10 倍，实际 loss 4.98（期望 3.31）

### 修复（第一层：仅修键名，不完整）

```python
if para not in self.u.keys():   # ✅ 用 para（持久对象）做键
    self.u[para] = 0
...
self.u[para] = ...
```

---

## 根因二：跨 step 计算图链导致内存泄漏（memory_check 失败的真因）

### 问题位置

`python/needle/optim.py`，`SGD.step()` 第 35 行

即便把字典键从 `grad` 改为 `para`，如下写法仍有致命问题：

```python
self.u[para] = ndl.broadcast_to(momentum * self.u[para], shape) + ...
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                同时读取和写入 self.u[para]！
```

### 计算图链的形成

每一步中，新的动量 tensor `u_new` 是通过对旧的 `u_old` 做 Tensor 算术计算得到的：

```
u_step32 = broadcast_to(momentum * u_step31, ...) + ...
                │
                ▼  (u_step32.op.inputs → ... → u_step31)
           u_step31
                │
                ▼  (u_step31.op.inputs → ... → u_step30)
           u_step30
                │
                ▼
              ...
                │
                ▼
           u_step1
```

**每一步都通过 autograd 计算图把新旧 momentum 串成了一条链。** 所有 32 个 step 的 momentum tensor **及其全部中间计算图节点**（每个 step 约 8 个 tensor：`broadcast_to` 结果、`momentum * u_old` 结果、`(1-momentum) * grad` 结果、`1-momentum` 结果、`momentum` 叶子、`ones` 叶子、`grad` 叶子等）全部存活。

### 为什么不改键名之前 memory_check 能过？

**关键区别**：原代码用 `grad` 做键时，每个 step 的 momentum tensor 存储在**不同的** `self.u[grad_i]` 条目中。虽然字典也在膨胀（多存了 N 个条目），但 `u_new = ... * self.u[old_grad] ...` 读到的是 **dict 中另一个 key 的值**。当旧字典条目最终被 GC 时（optimizer 生命周期结束），momentum tensor 链可以正常释放。而且旧版虽然也有计算图链，但 32 step × 4 param × ~8 tensor ≈ 1024 tensor 在 atol=1000 的容差边缘。

而修复版用 `para` 做键后，同一个 `self.u[para]` 先被读出参与计算，再被写入新值。**新值通过计算图直接引用旧值**，形成了一条贯穿所有 32 step 的不间断引用链。此时即使 `self.u` 只存了 4 个条目（每个参数一个），这 4 个条目却通过计算图牵着所有历史 step 的全部中间 tensor 不放。

### 张量泄漏计算

```
每个参数、每个 step 的存活 tensor：
  u_i, broadcast_to_result, (momentum * u_{i-1})_result,
  momentum_i 叶子, other_term, (1-momentum_i) 结果,
  ones_i 叶子, grad_i 叶子, Tensor(u_i) 包装
  ≈ 8-10 tensor/step

4 参数 × 32 step × ~9 tensor = 1152 tensor
× 5 个前置 SGD 测试 = 5760 tensor 仍然存活
```

`global_tensor_count()` 基线 ~387 + 5760 ≈ 4728 ≈ 实际 4720 ✓

---

## 修复方案

需要同时满足两个条件：
1. **字典键用 `para`**：保证动量状态跨 step 正确累积
2. **`.detach()` 切断计算图**：新旧 momentum 之间不能保留 autograd 引用

### 正确代码

```python
for para in self.params:
    shape = para.shape
    grad = para.grad + self.weight_decay * para
    if para not in self.u:
        self.u[para] = ndl.Tensor(np.zeros(shape), dtype='float32')
    momentum = Tensor([self.momentum])
    old_u = self.u[para].detach()  # ← 切断计算图链！
    self.u[para] = (ndl.broadcast_to(momentum * old_u, shape)
                    + (init.ones(*momentum.shape) - momentum) * grad)
    para.data = para.data - self.lr * Tensor(self.u[para], dtype='float32')
```

### 两个关键点

| 要点 | 说明 |
|------|------|
| `self.u[para]` 初始化为 Tensor | 不能用 Python `int` 0，因为 `int` 没有 `.detach()` 方法 |
| `old_u = self.u[para].detach()` | `.detach()` 返回一个与原 tensor 共享数据但切断梯度图的新 tensor。这样 `u_new` 的计算图不再引用 `u_old`，字典值被覆盖后 `u_old` 就能被 GC 回收 |

### 为什么 `.detach()` 不会改变数值正确性

`.detach()` 只是**切断 autograd 计算图**（将 `requires_grad` 标为 False 并创建一个无历史的 tensor），不影响 `.cached_data` 中的 numpy 数值。前向计算结果完全一致，但每一步的 momentum tensor 不再持有对上一步 tensor 的引用，内存可以在 step 间正常释放。


---
---

# DLSYS HW2 Adam + BatchNorm1d：`para.grad is None` 导致 `AddScalar(None)` 崩溃

## 问题背景

涉及的测试用例：
- `test_optim_adam_batchnorm_1`：Adam 优化器 + BatchNorm1d 组合测试

**错误信息**：
```
TypeError: unsupported operand type(s) for +: 'float' and 'NoneType'
```
发生在 `AddScalar.compute`（`ops_mathematic.py:33`）：`return a + self.scalar`，其中 `self.scalar` 为 `None`。

---

## 根因分析

这是一个**两层交互**导致的 bug。

### 第一层：BatchNorm1d.forward() 在 training 模式下"抛弃"了原来的 Parameter

**文件**：`python/needle/nn/nn_basic.py`，`BatchNorm1d.forward()` 第 189、192 行

```python
# __init__ 中：
self.running_mean = Parameter(array=init.zeros(dim))  # ← Parameter，会被 optimizer 收集
self.running_var  = Parameter(array=init.ones(dim))   # ← Parameter，会被 optimizer 收集

# forward() training 分支中：
self.running_mean = self.running_mean * (1 - self.momentum) + mean * self.momentum
self.running_var  = self.running_var  * (1 - self.momentum) + var  * self.momentum
```

这两行赋值语句将 `self.running_mean` / `self.running_var` **从 `Parameter` 对象替换为普通的 `Tensor`**（运算结果）。

**关键问题**：虽然 RHS 引用了原来的 `running_mean` / `running_var` Parameter，但计算结果（新的 plain Tensor）**并没有被模型输出所依赖**。模型的 forward 输出仅依赖 `x_minus_mean`、`var_broad_cast`、`self.weight`、`self.bias`。新的 `self.running_mean` / `self.running_var` 是一个"死胡同"——被计算出来但没有任何下游消费者。

```
计算图结构（training 模式）：

  x ─→ mean ─→ reshape_mean ─→ x_minus_mean ─┬─→ var ─→ ... ─→ output
  │                                            │                    ↑
  │                                            └─→ norm ───────────┘
  │                                                                 ↑
  └─→ (无关 output 的其他路径)                              weight, bias
      
  running_mean_old ─→ running_mean * (1-m) ─→ new_running_mean（死胡同，不连到 output）
                                               ↑
                              mean * m ───────┘（mean 参与了这个子图，但子图结果无人使用）
```

因此在 **第一次 forward+backward** 之后：

| optimizer 中的参数 | `para.grad` | 原因 |
|-------------------|-------------|------|
| weight (Linear 64→32) | Tensor ✓ | 在 loss → output 路径上 |
| bias (Linear 64→32) | Tensor ✓ | 在 loss → output 路径上 |
| weight (BatchNorm1d) | Tensor ✓ | 在 loss → output 路径上 |
| bias (BatchNorm1d) | Tensor ✓ | 在 loss → output 路径上 |
| **running_mean (旧 Parameter)** | **None ✗** | 死胡同子图，backward 的拓扑排序不会遍历到 |
| **running_var (旧 Parameter)** | **None ✗** | 死胡同子图，backward 的拓扑排序不会遍历到 |
| weight (Linear 32→16) | Tensor ✓ | 在 loss → output 路径上 |
| bias (Linear 32→16) | Tensor ✓ | 在 loss → output 路径上 |

```python
# backward() 中（autograd.py:380）：
reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))
# find_topo_sort 从 output 出发，沿 .inputs 方向做后序 DFS。
# "死胡同"节点（new_running_mean 等）不是任何 output 路径节点的 input，
# 因此永远不会被访问。它们的 .grad 保持 reset_grad() 设置的 None。
```

同时，`model.parameters()` 在 forward 之后只返回 6 个参数（原来的 running_mean/var Parameter 已从 `self.__dict__` 中被覆盖），但 optimizer 的 `self.params` 仍持有创建时缓存的 8 个参数引用——其中包括那两个 `.grad = None` 的孤儿 Parameter。

### 第二层：`para.grad is None` → Python `__radd__` 回退 → `AddScalar(None)`

**文件**：`python/needle/optim.py`，`Adam.step()` 第 79 行

```python
grad = para.grad + ndl.broadcast_to(Tensor([self.weight_decay], dtype='float32'), shape) * para
```

当 `para` 是已被抛弃的 `running_mean` 旧 Parameter 时，`para.grad` 为 `None`。Python 的运算符分派流程如下：

```
1. para.grad + mul_result
   → None + Tensor
   
2. None.__add__(Tensor) → 失败（NoneType 不支持此操作）
   
3. Python 回退到 Tensor.__radd__(None)
   由于 autograd.py:363: __radd__ = __add__
   
4. Tensor.__add__(self=mul_result, other=None):
     isinstance(None, Tensor) → False
     → AddScalar(None)(mul_result)    ← self.scalar = None！
   
5. AddScalar.compute(a):
     return a + self.scalar
     → numpy_array + None
     → TypeError: unsupported operand type(s) for +: 'float' and 'NoneType'
```

错误消息中的 `'float'` 是 numpy 数组中元素的类型，`'NoneType'` 是 `self.scalar` 的类型。

---

## 修复方案

需要同时修改 `BatchNorm1d.__init__` 和 `BatchNorm1d.forward`。

### 修改 1：`__init__` — running_mean/var 不应是 Parameter

**文件**：`python/needle/nn/nn_basic.py`，`BatchNorm1d.__init__`

```python
# 错误 ❌：running_mean/var 是 Parameter，会被 optimizer 收集
self.running_mean = Parameter(array=init.zeros(dim))
self.running_var  = Parameter(array=init.ones(dim))

# 正确 ✅：running_mean/var 是普通 Tensor（requires_grad=False）
# init.zeros/ones 默认 requires_grad=False，不会被 _unpack_params 收集
self.running_mean = init.zeros(dim)
self.running_var  = init.ones(dim)
```

**原理**：`_unpack_params` 只收集 `Parameter` 实例。普通 `Tensor`（即使 `requires_grad=True`）也不会被加入 `model.parameters()`。`running_mean` 和 `running_var` 在语义上是推理时使用的**统计量**（通过 EMA 更新），**不应参与梯度下降优化**，因此本来就不该是 `Parameter`。

### 修改 2：`forward` — 用 `.data` 原地更新，不构建计算图

**文件**：`python/needle/nn/nn_basic.py`，`BatchNorm1d.forward` training 分支

```python
# 错误 ❌：构建计算图节点（且结果不连到 output），每次 forward 创建孤立子图
self.running_mean = self.running_mean * (1 - self.momentum) + mean * self.momentum
self.running_var  = self.running_var  * (1 - self.momentum) + var  * self.momentum

# 正确 ✅：用 .data（cached_data 级别）做原地更新，完全脱离计算图
self.running_mean.data = (
    (1 - self.momentum) * self.running_mean.realize_cached_data()
    + self.momentum * mean.realize_cached_data()
)
self.running_var.data = (
    (1 - self.momentum) * self.running_var.realize_cached_data()
    + self.momentum * var.realize_cached_data()
)
```

**原理**：
1. `.realize_cached_data()` 取出底层 numpy 数组
2. 纯 numpy 运算不产生任何 autograd 节点
3. `.data = ...`（即 `cached_data` setter）原地替换数据，不改变 Tensor 对象身份

这样 `self.running_mean` / `self.running_var` **始终是同一个 Tensor 对象**，`.data` 的值被原地更新，不创建新 Tensor、不构建计算图、不产生死胡同节点。

### 为什么两个修改缺一不可

| 只做修改 1 | 只做修改 2 |
|-----------|-----------|
| running_mean/var 不会进入 optimizer，step() 不报错 | running_mean/var 仍是 Parameter，optimizer 仍持有引用 |
| 但 forward 每次创建孤立子图，epoch 多了会内存泄漏 | forward 不泄漏图节点 |
| 短期测试能过，长期有问题 | step() 仍然对它们调用 `para.grad + ...`（grad=None）→ 崩溃 |
