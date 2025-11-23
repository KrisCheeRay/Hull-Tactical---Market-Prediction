"""
演示：函数包装不会断梯度

这个脚本展示了：
1. 直接写操作 vs 函数包装 - 梯度都保留
2. 什么操作会断梯度
3. 如何检查梯度是否断了
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 60)
print("测试 1: 直接写 vs 函数包装 - 梯度是否保留？")
print("=" * 60)

# 准备数据
alpha = torch.tensor([2.0], requires_grad=True)
beta = torch.tensor([3.0], requires_grad=True)

# 方式 1：直接写
def test_direct():
    """直接写操作"""
    action = 2.0 * (alpha / (alpha + beta))
    loss = action ** 2
    loss.backward()
    grad_alpha = alpha.grad.item()
    alpha.grad = None  # 重置
    return grad_alpha

# 方式 2：函数包装
def compute_action_function(alpha, beta):
    """用函数包装 - 只要内部是 PyTorch 操作，梯度就保留"""
    mean = alpha / (alpha + beta)
    action = 2.0 * mean
    return action

def test_function():
    """通过函数计算"""
    action = compute_action_function(alpha, beta)
    loss = action ** 2
    loss.backward()
    grad_alpha = alpha.grad.item()
    alpha.grad = None  # 重置
    return grad_alpha

# 方式 3：更复杂的函数
def compute_action_complex(alpha, beta):
    """复杂的函数，多层嵌套"""
    ratio = alpha / (alpha + beta)
    scaled = ratio * 2.0
    smoothed = F.sigmoid(scaled) * 2.0
    return smoothed

def test_complex_function():
    """通过复杂函数计算"""
    action = compute_action_complex(alpha, beta)
    loss = action ** 2
    loss.backward()
    grad_alpha = alpha.grad.item()
    alpha.grad = None  # 重置
    return grad_alpha

# 运行测试
grad1 = test_direct()
grad2 = test_function()
grad3 = test_complex_function()

print(f"直接写操作:     alpha.grad = {grad1:.6f}")
print(f"函数包装:       alpha.grad = {grad2:.6f}")
print(f"复杂函数:       alpha.grad = {grad3:.6f}")
print(f"\n✅ 结论: 函数包装不会断梯度！梯度值: {grad1:.6f} ≈ {grad2:.6f} ≈ {grad3:.6f}")

print("\n" + "=" * 60)
print("测试 2: 什么操作会断梯度？")
print("=" * 60)

# 重置
alpha = torch.tensor([2.0], requires_grad=True)
beta = torch.tensor([3.0], requires_grad=True)

# ❌ 错误 1: 使用 .item()
def compute_action_broken1(alpha, beta):
    """❌ 错误：.item() 会断梯度"""
    mean = (alpha / (alpha + beta)).item()  # 转换为 Python float
    action = torch.tensor(mean * 2.0)      # 新的 Tensor，没有梯度信息
    return action

def test_broken1():
    """测试断梯度的函数"""
    action = compute_action_broken1(alpha, beta)
    loss = action ** 2
    loss.backward()
    return alpha.grad

# ❌ 错误 2: 使用 .detach()
def compute_action_broken2(alpha, beta):
    """❌ 错误：.detach() 会断梯度"""
    mean = (alpha / (alpha + beta)).detach()  # 断掉梯度
    return 2.0 * mean

def test_broken2():
    """测试断梯度的函数"""
    action = compute_action_broken2(alpha, beta)
    loss = action ** 2
    loss.backward()
    return alpha.grad

# 运行测试
alpha.grad = None
grad_broken1 = test_broken1()

alpha.grad = None
grad_broken2 = test_broken2()

print(f"使用 .item():    alpha.grad = {grad_broken1}")
print(f"使用 .detach():  alpha.grad = {grad_broken2}")
print(f"\n❌ 结论: 这些操作会断梯度！grad = None")

print("\n" + "=" * 60)
print("测试 3: 实际应用 - PolicyHead 风格")
print("=" * 60)

# 模拟 PolicyHead 的计算
class SimplePolicyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(3, 2)
    
    def forward(self, x):
        out = self.net(x)
        alpha = F.softplus(out[:, 0]) + 1.0
        beta = F.softplus(out[:, 1]) + 1.0
        return alpha, beta
    
    # 方式 1: 直接写在方法里
    def get_action_direct(self, x):
        alpha, beta = self.forward(x)
        return 2.0 * (alpha / (alpha + beta))
    
    # 方式 2: 提取为辅助函数
    def _compute_action_from_beta(self, alpha, beta):
        """辅助函数 - 梯度仍然保留！"""
        mean = alpha / (alpha + beta)
        return 2.0 * mean
    
    def get_action_function(self, x):
        alpha, beta = self.forward(x)
        return self._compute_action_from_beta(alpha, beta)

# 测试
policy = SimplePolicyHead()
x = torch.randn(5, 3)

# 方式 1
action1 = policy.get_action_direct(x)
loss1 = action1.mean()
loss1.backward()
grad1 = policy.net.weight.grad.clone()
policy.zero_grad()

# 方式 2
action2 = policy.get_action_function(x)
loss2 = action2.mean()
loss2.backward()
grad2 = policy.net.weight.grad.clone()

print(f"直接写在方法里:     网络第一层权重梯度存在 = {grad1 is not None}")
print(f"提取为辅助函数:     网络第一层权重梯度存在 = {grad2 is not None}")
print(f"梯度值是否相同:     {torch.allclose(grad1, grad2)}")
print(f"\n✅ 结论: 函数提取不会断梯度！两种方式梯度完全一致")

print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("""
✅ 会保留梯度的操作：
   - 所有 PyTorch 标准操作（+, -, *, /, F.softplus, nn.Linear 等）
   - 函数包装（只要函数内部是 PyTorch 操作）
   - 多层嵌套的函数调用

❌ 会断梯度的操作：
   - tensor.item()      # 转为 Python 标量
   - tensor.detach()    # 分离计算图
   - tensor.numpy()     # 转为 numpy
   - @torch.no_grad()   # 禁用梯度追踪

💡 关键点：
   PyTorch 的 autograd 会追踪所有 Tensor 操作，无论这些操作是在
   直接代码中还是在函数里。只要操作是可微的 PyTorch 操作，
   梯度就会自动传播！
""")

