"""
Demo 5: 断点回归 (Regression Discontinuity Design, RDD)
========================================================
论文: Imbens & Lemieux (2008)

核心思想:
  当处理由一个"运行变量"(running variable)超过阈值决定时,
  阈值两侧的个体几乎相同 (局部随机化),
  因此阈值处的结果跳变就是因果效应.

经典案例: 考试成绩 >= 60分获得奖学金, 研究奖学金对未来收入的影响
  59分和60分的学生本质差异极小, 但一个有奖学金一个没有.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

np.random.seed(42)
os.makedirs("output", exist_ok=True)

n = 2000
CUTOFF = 60
TRUE_EFFECT = 5.0  # 奖学金的真实效应

# ============================================================
# 1. 数据生成
# ============================================================
score = np.random.normal(60, 15, n)  # 考试分数 (运行变量)
treatment = (score >= CUTOFF).astype(int)

# 结果: 未来收入 (千元/月)
# 非线性基线关系 + 阈值处的跳变
outcome = (8 + 0.15 * score + 0.002 * (score - 60) ** 2 +
           TRUE_EFFECT * treatment + np.random.normal(0, 2, n))

df = pd.DataFrame({"score": score, "treatment": treatment, "outcome": outcome})

print("=" * 60)
print("Demo 5: 断点回归 (RDD)")
print("=" * 60)
print(f"真实效应: {TRUE_EFFECT}")
print(f"阈值: 分数 >= {CUTOFF} 获得奖学金")

# ============================================================
# 2. Naive 比较 (错误)
# ============================================================
naive = df[df.treatment == 1]["outcome"].mean() - df[df.treatment == 0]["outcome"].mean()
print(f"\nNaive 比较: {naive:.2f}  (有偏! 分数高本身就意味着更好的结果)")

# ============================================================
# 3. Sharp RDD: 局部线性回归
# ============================================================
bandwidths = [5, 10, 15, 20]
print(f"\n--- 不同带宽的 RDD 估计 ---")
rdd_results = {}
for bw in bandwidths:
    local = df[(df.score >= CUTOFF - bw) & (df.score <= CUTOFF + bw)].copy()
    local["score_centered"] = local["score"] - CUTOFF

    reg = LinearRegression()
    X = np.column_stack([local["score_centered"], local["treatment"],
                         local["score_centered"] * local["treatment"]])
    reg.fit(X, local["outcome"])
    rdd_effect = reg.coef_[1]
    rdd_results[bw] = rdd_effect
    print(f"  带宽 h={bw:2d}: RDD效应 = {rdd_effect:.2f}  (N={len(local)})")

# ============================================================
# 4. McCrary 密度检验 (检查操纵)
# ============================================================
bins_left = np.sum((score >= CUTOFF - 5) & (score < CUTOFF))
bins_right = np.sum((score >= CUTOFF) & (score < CUTOFF + 5))
print(f"\nMcCrary 密度检验 (是否有人操纵分数?):")
print(f"  阈值左侧 [{CUTOFF-5}, {CUTOFF}): {bins_left}")
print(f"  阈值右侧 [{CUTOFF}, {CUTOFF+5}): {bins_right}")
print(f"  → 密度比 = {bins_right / bins_left:.2f}  (接近1 = 无操纵)")

# ============================================================
# 5. 协变量连续性检验
# ============================================================
df["female"] = np.random.binomial(1, 0.5 + 0.001 * (score - 60), n)
local_check = df[(df.score >= 55) & (df.score <= 65)]
left_female = local_check[local_check.score < 60]["female"].mean()
right_female = local_check[local_check.score >= 60]["female"].mean()
print(f"\n协变量连续性检验 (性别比例):")
print(f"  阈值左侧: {left_female:.3f}")
print(f"  阈值右侧: {right_female:.3f}")
print(f"  → 差异 = {abs(right_female - left_female):.3f}  (接近0 = 通过)")

# ============================================================
# 6. 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) 散点图 + RDD
axes[0].scatter(df[df.treatment == 0]["score"], df[df.treatment == 0]["outcome"],
                alpha=0.15, s=10, color="blue", label="No Scholarship")
axes[0].scatter(df[df.treatment == 1]["score"], df[df.treatment == 1]["outcome"],
                alpha=0.15, s=10, color="red", label="Scholarship")

# 局部多项式拟合
for side, color in [(0, "blue"), (1, "red")]:
    sub = df[df.treatment == side].sort_values("score")
    z = np.polyfit(sub.score, sub.outcome, 2)
    p = np.poly1d(z)
    x_line = np.linspace(sub.score.min(), sub.score.max(), 100)
    axes[0].plot(x_line, p(x_line), color=color, linewidth=2)

axes[0].axvline(CUTOFF, color="black", linestyle="--", linewidth=2, label=f"Cutoff = {CUTOFF}")
axes[0].set_xlabel("Test Score (Running Variable)")
axes[0].set_ylabel("Monthly Income (k)")
axes[0].set_title(f"RDD: Effect = {rdd_results[10]:.2f}")
axes[0].legend()

# (b) 局部放大 (带宽=10)
bw = 10
local = df[(df.score >= CUTOFF - bw) & (df.score <= CUTOFF + bw)]
axes[1].scatter(local[local.treatment == 0]["score"],
                local[local.treatment == 0]["outcome"],
                alpha=0.3, s=15, color="blue")
axes[1].scatter(local[local.treatment == 1]["score"],
                local[local.treatment == 1]["outcome"],
                alpha=0.3, s=15, color="red")

for side, color in [(0, "blue"), (1, "red")]:
    sub = local[local.treatment == side].sort_values("score")
    z = np.polyfit(sub.score, sub.outcome, 1)
    p = np.poly1d(z)
    x_line = np.linspace(sub.score.min(), sub.score.max(), 50)
    axes[1].plot(x_line, p(x_line), color=color, linewidth=2)

axes[1].axvline(CUTOFF, color="black", linestyle="--", linewidth=2)
axes[1].set_xlabel("Test Score")
axes[1].set_ylabel("Monthly Income (k)")
axes[1].set_title(f"Zoomed In (bandwidth = {bw})")

# (c) 带宽敏感性分析
bws = list(range(3, 25))
effects = []
for bw in bws:
    loc = df[(df.score >= CUTOFF - bw) & (df.score <= CUTOFF + bw)].copy()
    loc["sc"] = loc["score"] - CUTOFF
    reg = LinearRegression()
    X = np.column_stack([loc["sc"], loc["treatment"], loc["sc"] * loc["treatment"]])
    reg.fit(X, loc["outcome"])
    effects.append(reg.coef_[1])

axes[2].plot(bws, effects, "ko-", markersize=4)
axes[2].axhline(TRUE_EFFECT, color="green", linestyle="--", linewidth=2, label=f"True = {TRUE_EFFECT}")
axes[2].set_xlabel("Bandwidth")
axes[2].set_ylabel("RDD Effect Estimate")
axes[2].set_title("Bandwidth Sensitivity Analysis")
axes[2].legend()

plt.tight_layout()
plt.savefig("output/05_rdd.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n图表已保存: output/05_rdd.png")

print(f"""
{'=' * 60}
总结:
  真实效应:   {TRUE_EFFECT:.2f}
  Naive:      {naive:.2f}   ← 偏大, 分数本身影响了结果
  RDD(h=10):  {rdd_results[10]:.2f}   ← 接近真值!

RDD 的关键:
  1. 阈值处的"局部随机化" — 59分和60分的人本质相同
  2. 需要验证: 无操纵 (McCrary检验), 协变量连续
  3. 带宽选择很重要: 太窄方差大, 太宽偏差大
{'=' * 60}
""")
