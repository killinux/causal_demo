"""
Demo 1: ATE (平均处理效应) 估计 + Simpson's Paradox
=====================================================
核心问题: 为什么简单比较"吃药组"和"不吃药组"的均值会得到错误结论?

原理:
- 真实的因果效应 = E[Y(1) - Y(0)]  (每个人吃药vs不吃药的差异的期望)
- 但我们观测到的差异 = E[Y|T=1] - E[Y|T=0]  (这包含了选择偏差!)
- 选择偏差: 重症患者更可能吃药, 但重症本身导致结果更差
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

np.random.seed(42)
os.makedirs("output", exist_ok=True)

# ============================================================
# 1. 数据生成: 模拟一个药物试验
# ============================================================
n = 2000

# X: 病情严重程度 (混杂因子, 同时影响是否用药和康复)
severity = np.random.normal(5, 2, n)

# T: 是否接受治疗 (病情越重, 越可能接受治疗 — 这就是选择偏差的来源)
prob_treatment = 1 / (1 + np.exp(-(severity - 5)))  # sigmoid
treatment = np.random.binomial(1, prob_treatment)

# Y: 康复天数 (真实因果效应: 吃药减少3天)
TRUE_ATE = -3.0
recovery_days = 10 + 2 * severity + TRUE_ATE * treatment + np.random.normal(0, 1, n)

df = pd.DataFrame({
    "severity": severity,
    "treatment": treatment,
    "recovery_days": recovery_days
})

print("=" * 60)
print("Demo 1: ATE 估计与 Simpson's Paradox")
print("=" * 60)
print(f"\n真实因果效应 (ATE): {TRUE_ATE} 天 (吃药减少3天康复时间)")

# ============================================================
# 2. 错误方法: 简单比较均值 (Naive Comparison)
# ============================================================
naive_ate = (df[df.treatment == 1]["recovery_days"].mean() -
             df[df.treatment == 0]["recovery_days"].mean())
print(f"\n--- 方法1: 简单比较均值 (有偏) ---")
print(f"Naive ATE = {naive_ate:.2f} 天")
print(f"→ 结论错误! 因为重症患者更可能吃药, 混入了严重程度的影响")

# ============================================================
# 3. 正确方法1: 条件均值 (控制混杂因子)
# ============================================================
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(df[["severity", "treatment"]], df["recovery_days"])
conditional_ate = reg.coef_[1]
print(f"\n--- 方法2: 线性回归控制混杂因子 ---")
print(f"Conditional ATE = {conditional_ate:.2f} 天")
print(f"→ 接近真实值 {TRUE_ATE}!")

# ============================================================
# 4. 正确方法2: 分层估计 (Stratification)
# ============================================================
df["severity_group"] = pd.qcut(df["severity"], q=5, labels=False)
stratum_ates = []
for g in range(5):
    sub = df[df.severity_group == g]
    if sub.treatment.sum() > 0 and (sub.treatment == 0).sum() > 0:
        ate_g = (sub[sub.treatment == 1]["recovery_days"].mean() -
                 sub[sub.treatment == 0]["recovery_days"].mean())
        stratum_ates.append((len(sub), ate_g))

stratified_ate = sum(n_g * ate_g for n_g, ate_g in stratum_ates) / sum(n_g for n_g, _ in stratum_ates)
print(f"\n--- 方法3: 分层估计 ---")
print(f"Stratified ATE = {stratified_ate:.2f} 天")
print(f"→ 也接近真实值!")

# ============================================================
# 5. 可视化: Simpson's Paradox
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) 整体看: 吃药组康复更慢 (错误印象)
axes[0].hist(df[df.treatment == 0]["recovery_days"], bins=30, alpha=0.6, label="No Treatment", color="blue")
axes[0].hist(df[df.treatment == 1]["recovery_days"], bins=30, alpha=0.6, label="Treatment", color="red")
axes[0].axvline(df[df.treatment == 0]["recovery_days"].mean(), color="blue", linestyle="--", linewidth=2)
axes[0].axvline(df[df.treatment == 1]["recovery_days"].mean(), color="red", linestyle="--", linewidth=2)
axes[0].set_title(f"Naive Comparison\nATE = {naive_ate:.2f} (BIASED!)", fontsize=13)
axes[0].set_xlabel("Recovery Days")
axes[0].legend()

# (b) 散点图: 混杂因子的影响
colors = ["blue" if t == 0 else "red" for t in df.treatment]
axes[1].scatter(df.severity, df.recovery_days, c=colors, alpha=0.3, s=10)
axes[1].set_xlabel("Severity (Confounder)")
axes[1].set_ylabel("Recovery Days")
axes[1].set_title("Confounder Effect\nSeverity → Treatment & Recovery", fontsize=13)
from matplotlib.patches import Patch
axes[1].legend(handles=[Patch(color="blue", label="No Treatment"), Patch(color="red", label="Treatment")])

# (c) 分层后: 每层内吃药组更快 (正确结论)
for g in range(5):
    sub = df[df.severity_group == g]
    y0 = sub[sub.treatment == 0]["recovery_days"].mean()
    y1 = sub[sub.treatment == 1]["recovery_days"].mean()
    axes[2].bar(g - 0.15, y0, width=0.3, color="blue", alpha=0.7)
    axes[2].bar(g + 0.15, y1, width=0.3, color="red", alpha=0.7)

axes[2].set_xlabel("Severity Group (0=mild, 4=severe)")
axes[2].set_ylabel("Avg Recovery Days")
axes[2].set_title(f"Stratified Comparison\nATE = {stratified_ate:.2f} (CORRECT!)", fontsize=13)
axes[2].legend(handles=[Patch(color="blue", label="No Treatment"), Patch(color="red", label="Treatment")])

plt.tight_layout()
plt.savefig("output/01_simpson_paradox.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n图表已保存: output/01_simpson_paradox.png")

# ============================================================
# 6. 总结
# ============================================================
print(f"""
{'=' * 60}
总结:
  真实 ATE:      {TRUE_ATE:.2f}
  Naive (有偏):  {naive_ate:.2f}  ← 因为混杂因子, 方向都搞反了!
  回归 (控制):   {conditional_ate:.2f}  ← 控制混杂后, 接近真值
  分层估计:      {stratified_ate:.2f}  ← 分层后, 也接近真值

关键教训:
  当存在混杂因子时, 简单比较均值会产生 Simpson's Paradox,
  必须通过某种方式"控制"混杂因子才能得到正确的因果效应估计.
{'=' * 60}
""")
