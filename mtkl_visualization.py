import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置随机种子以确保可重复
np.random.seed(42)

# 生成 1000 个独立的高斯随机变量 a, b, c ~ N(0,1)
a = np.random.normal(0, 1, 1000)
b = np.random.normal(0, 1, 1000)
c = np.random.normal(0, 1, 1000)

# 计算 x, y, w, z
x = a + b + c
y = a + b - c
w = a - b + c
z = a - b - c

# 筛选满足 x > 0, y > 0, w > 0 的点
mask = (x > 0) & (y > 0) & (w > 0)

# 在满足条件的点中，用 z 的正负分类颜色
a_sel = a[mask]
b_sel = b[mask]
c_sel = c[mask]
z_sel = z[mask]
l0 = z_sel > 0

# 输出满足条件的样本数与 z > 0 的比例
sample_count = np.sum(mask)
z_pos_prob = np.sum(l0) / sample_count

# 准备颜色
colors = np.where(l0, 'red', 'blue')

# 3D 可视化
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(a_sel, b_sel, c_sel, c=colors, label='Samples')

# 添加 z=0 的边界平面：a - b - c = 0，即 c = a - b
aa, bb = np.meshgrid(np.linspace(-3, 3, 10), np.linspace(-3, 3, 10))
cc = aa - bb
ax.plot_surface(aa, bb, cc, alpha=0.3, color='gray', label='z=0 boundary')

ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_zlabel('c')
ax.set_title(f'Z>0 Probability: {z_pos_prob:.3f} ({sample_count} samples)')
plt.tight_layout()
plt.savefig("3d.png")
