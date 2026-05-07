"""
Demo 6: DAG + do-演算 (Pearl 因果图方法)
=========================================
论文: Pearl (1995) "Causal Diagrams for Empirical Research"
书籍: Pearl (2009) "Causality"

核心思想:
  用有向无环图 (DAG) 表示变量间的因果关系,
  通过 d-分离 判断条件独立性,
  通过 do-演算 将干预 P(Y|do(X)) 转化为可从观测数据估计的量.

三条核心规则:
  1. 后门准则 (Back-door Criterion): 控制哪些变量可以消除混杂
  2. 前门准则 (Front-door Criterion): 即使有不可观测混杂, 有时仍可识别
  3. do-演算三条规则: 最一般的因果效应识别方法
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

np.random.seed(42)
os.makedirs("output", exist_ok=True)

print("=" * 60)
print("Demo 6: DAG + do-演算")
print("=" * 60)

# ============================================================
# 1. 后门准则 (Back-door Criterion)
# ============================================================
# DAG: Z → X → Y, Z → Y (Z是混杂因子)
print("\n--- 示例1: 后门准则 ---")
print("DAG: Z → X → Y, Z → Y")
print("Z是混杂因子, 控制Z后可识别X对Y的因果效应")

n = 5000
Z = np.random.normal(0, 1, n)
X = 0.8 * Z + np.random.normal(0, 1, n)
Y = 2.0 * X + 1.5 * Z + np.random.normal(0, 1, n)  # 真实效应 = 2.0

from scipy import stats

# 不控制Z (有偏)
slope_biased, _, _, _, _ = stats.linregress(X, Y)
print(f"  不控制Z: β = {slope_biased:.3f} (有偏, 真值=2.0)")

# 控制Z (后门准则)
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(np.column_stack([X, Z]), Y)
print(f"  控制Z:   β = {reg.coef_[0]:.3f} (后门准则, 接近真值)")

# ============================================================
# 2. 前门准则 (Front-door Criterion)
# ============================================================
# DAG: U → X, U → Y, X → M → Y (U不可观测!)
print("\n--- 示例2: 前门准则 ---")
print("DAG: U → X, U → Y, X → M → Y")
print("U不可观测, 但通过中介M仍可识别效应!")

TRUE_X_M = 1.5  # X → M 的效应
TRUE_M_Y = 2.0  # M → Y 的效应
TRUE_TOTAL = TRUE_X_M * TRUE_M_Y  # 总因果效应 = 3.0

U = np.random.normal(0, 1, n)
X_fd = 0.7 * U + np.random.normal(0, 1, n)
M = TRUE_X_M * X_fd + np.random.normal(0, 0.5, n)  # X → M
Y_fd = TRUE_M_Y * M + 1.0 * U + np.random.normal(0, 0.5, n)  # M → Y, U → Y

# Naive (有偏, 因为U)
slope_naive, _, _, _, _ = stats.linregress(X_fd, Y_fd)
print(f"  Naive:    β = {slope_naive:.3f} (有偏, 真值={TRUE_TOTAL:.1f})")

# 前门准则: 分两步
# Step 1: X → M (无混杂, 因为U不影响M除了通过X)
slope_xm, _, _, _, _ = stats.linregress(X_fd, M)
# Step 2: M → Y (控制X, 因为X是M-Y路径中U的后门)
reg_my = LinearRegression().fit(np.column_stack([M, X_fd]), Y_fd)
slope_my = reg_my.coef_[0]
# 总效应
frontdoor_effect = slope_xm * slope_my
print(f"  前门准则: β = {frontdoor_effect:.3f} (X→M={slope_xm:.2f} × M→Y={slope_my:.2f})")

# ============================================================
# 3. 碰撞节点 (Collider Bias)
# ============================================================
print("\n--- 示例3: 碰撞节点偏差 ---")
print("DAG: X → C ← Y (C是碰撞节点)")
print("X和Y本来独立, 但控制C后产生虚假相关!")

X_col = np.random.normal(0, 1, n)
Y_col = np.random.normal(0, 1, n)  # X和Y独立!
C = X_col + Y_col + np.random.normal(0, 0.5, n)  # C是碰撞节点

# 不控制C: X和Y独立
r_no_control, p_no = stats.pearsonr(X_col, Y_col)
print(f"  不控制C: r = {r_no_control:.3f}, p = {p_no:.3f} (正确: 独立)")

# 控制C: 产生虚假相关! (Berkson's Paradox)
reg_c = LinearRegression().fit(C.reshape(-1, 1), Y_col)
Y_resid = Y_col - reg_c.predict(C.reshape(-1, 1))
reg_c2 = LinearRegression().fit(C.reshape(-1, 1), X_col)
X_resid = X_col - reg_c2.predict(C.reshape(-1, 1))
r_control, p_ctrl = stats.pearsonr(X_resid, Y_resid)
print(f"  控制C:   r = {r_control:.3f}, p = {p_ctrl:.6f} (错误! 虚假相关)")
print(f"  → 不该控制碰撞节点!")

# ============================================================
# 4. d-分离规则总结
# ============================================================
print(f"""
--- d-分离规则 ---
  链 (Chain):     X → M → Y    控制M → X和Y独立 (阻断)
  叉 (Fork):      X ← Z → Y    控制Z → X和Y独立 (阻断混杂)
  碰撞 (Collider): X → C ← Y   不控制C → X和Y独立
                                控制C → 产生虚假相关! (打开路径)
""")

# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

def draw_node(ax, x, y, text, color="lightyellow"):
    ax.add_patch(mpatches.FancyBboxPatch((x - 0.4, y - 0.3), 0.8, 0.6,
                                          boxstyle="round,pad=0.1",
                                          facecolor=color, edgecolor="black", linewidth=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=13, fontweight="bold")

def draw_arrow(ax, x1, y1, x2, y2, color="black", style="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2, linestyle=style))

# (a) 后门准则
ax = axes[0]
ax.set_xlim(-1, 5)
ax.set_ylim(-1, 4)
draw_node(ax, 0, 3, "Z", "lightsalmon")
draw_node(ax, 0, 1, "X", "lightyellow")
draw_node(ax, 4, 1, "Y", "lightgreen")
draw_arrow(ax, 0, 2.7, 0, 1.3)      # Z → X
draw_arrow(ax, 0.4, 3, 3.6, 1.3)    # Z → Y
draw_arrow(ax, 0.4, 1, 3.6, 1)      # X → Y
ax.text(2, -0.5, "Back-door:\nControl Z to block Z→X→...→Y\nvia the backdoor path Z→Y",
        ha="center", fontsize=10, style="italic")
ax.set_title("Back-door Criterion", fontsize=14)
ax.axis("off")

# (b) 前门准则
ax = axes[1]
ax.set_xlim(-1, 5)
ax.set_ylim(-1, 4)
draw_node(ax, 2, 3.5, "U", "lightsalmon")
draw_node(ax, 0, 1, "X", "lightyellow")
draw_node(ax, 2, 1, "M", "lightblue")
draw_node(ax, 4, 1, "Y", "lightgreen")
draw_arrow(ax, 1.6, 3.2, 0.4, 1.3, "red", "--")   # U → X
draw_arrow(ax, 2.4, 3.2, 3.6, 1.3, "red", "--")   # U → Y
draw_arrow(ax, 0.4, 1, 1.6, 1)      # X → M
draw_arrow(ax, 2.4, 1, 3.6, 1)      # M → Y
ax.text(2, -0.5, "Front-door:\nU unobserved, but X→M→Y\nidentifies the causal effect!",
        ha="center", fontsize=10, style="italic")
ax.set_title("Front-door Criterion", fontsize=14)
ax.axis("off")

# (c) 碰撞节点
ax = axes[2]
ax.set_xlim(-1, 5)
ax.set_ylim(-1, 4)
draw_node(ax, 0, 3, "X", "lightyellow")
draw_node(ax, 4, 3, "Y", "lightgreen")
draw_node(ax, 2, 1, "C", "lightsalmon")
draw_arrow(ax, 0.4, 2.7, 1.6, 1.3)   # X → C
draw_arrow(ax, 3.6, 2.7, 2.4, 1.3)   # Y → C
ax.text(2, -0.5, "Collider Bias:\nX and Y are independent!\nControlling C creates spurious correlation",
        ha="center", fontsize=10, style="italic")
ax.set_title("Collider (DON'T control!)", fontsize=14)
ax.axis("off")

plt.tight_layout()
plt.savefig("output/06_dag.png", dpi=150, bbox_inches="tight")
plt.close()
print("图表已保存: output/06_dag.png")

print(f"""
{'=' * 60}
总结:
  后门准则: 控制Z后, X→Y因果效应 = 2.000 (估计={reg.coef_[0]:.3f})
  前门准则: 通过中介M, 总效应 = {TRUE_TOTAL:.1f} (估计={frontdoor_effect:.3f})
  碰撞偏差: 不该控制碰撞节点, 否则产生虚假关联

核心规则:
  - 控制混杂因子 (fork) ✓
  - 不控制碰撞节点 (collider) ✓
  - 不控制中介变量 (除非你只想要直接效应) ✓
{'=' * 60}
""")
