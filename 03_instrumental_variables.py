"""
Demo 3: 工具变量 (Instrumental Variables, IV)
==============================================
论文: Angrist & Imbens (1994) — 2021年诺贝尔经济学奖

核心思想:
  当存在不可观测的混杂因子时, OLS 估计有偏.
  工具变量 Z 满足三个条件:
    1. 相关性: Z 与处理 T 相关 (Cov(Z,T) != 0)
    2. 排他性: Z 只通过 T 影响 Y (Z → T → Y, 但 Z ↛ Y)
    3. 独立性: Z 与未观测混杂 U 独立

经典案例: 教育回报
  - Y = 收入, T = 受教育年限, U = 天赋(不可观测)
  - Z = 出生季度 (影响入学年龄→教育年限, 但不直接影响收入)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import os

np.random.seed(42)
os.makedirs("output", exist_ok=True)

n = 5000
TRUE_BETA = 0.8  # 教育每多一年, 收入增加 0.8 (对数)

# ============================================================
# 1. 数据生成
# ============================================================
# U: 不可观测的混杂因子 (天赋/家庭背景)
U = np.random.normal(0, 1, n)

# Z: 工具变量 (出生在上半年=1, 下半年=0 — 影响入学年龄)
Z = np.random.binomial(1, 0.5, n)

# T: 受教育年限 (受 Z 和 U 影响)
education = 12 + 2 * Z + 1.5 * U + np.random.normal(0, 1, n)

# Y: 对数收入 (受教育和天赋影响, 但 Z 不直接影响)
log_income = 8 + TRUE_BETA * education + 1.0 * U + np.random.normal(0, 0.5, n)

df = pd.DataFrame({
    "Z": Z, "education": education,
    "log_income": log_income, "U": U
})

print("=" * 60)
print("Demo 3: 工具变量 (IV)")
print("=" * 60)
print(f"真实因果效应 β = {TRUE_BETA}")

# ============================================================
# 2. OLS 估计 (有偏 — 遗漏了天赋 U)
# ============================================================
slope_ols, intercept_ols, r, p, se = stats.linregress(education, log_income)
print(f"\nOLS 估计: β = {slope_ols:.4f}  (有偏! 因为遗漏了天赋U)")
print(f"  → 向上偏: 天赋高的人教育也高, OLS把天赋的效果也算在教育上了")

# ============================================================
# 3. IV / 2SLS 估计 (两阶段最小二乘)
# ============================================================
# 第一阶段: education = π0 + π1 * Z + v
slope_1st, intercept_1st, _, _, _ = stats.linregress(Z, education)
education_hat = intercept_1st + slope_1st * Z

# 第二阶段: log_income = β0 + β1 * education_hat + e
slope_iv, intercept_iv, _, _, _ = stats.linregress(education_hat, log_income)
print(f"\nIV/2SLS 估计:")
print(f"  第一阶段: education = {intercept_1st:.2f} + {slope_1st:.2f} * Z")
print(f"  F-statistic (第一阶段): {(slope_1st / (1/np.sqrt(n)))**2:.1f}  (>10 = 强工具变量)")
print(f"  第二阶段: β = {slope_iv:.4f}  ← 接近真值 {TRUE_BETA}!")

# ============================================================
# 4. Wald 估计量 (简化版 IV)
# ============================================================
wald = ((df[df.Z == 1]["log_income"].mean() - df[df.Z == 0]["log_income"].mean()) /
        (df[df.Z == 1]["education"].mean() - df[df.Z == 0]["education"].mean()))
print(f"\nWald 估计: β = {wald:.4f}")
print(f"  Wald = [E(Y|Z=1)-E(Y|Z=0)] / [E(T|Z=1)-E(T|Z=0)]")

# ============================================================
# 5. 弱工具变量问题演示
# ============================================================
print(f"\n--- 弱工具变量对比 ---")
weak_results = []
for z_strength in [0.1, 0.5, 1.0, 2.0, 5.0]:
    edu_weak = 12 + z_strength * Z + 1.5 * U + np.random.normal(0, 1, n)
    s1, i1, _, _, _ = stats.linregress(Z, edu_weak)
    edu_hat_w = i1 + s1 * Z
    s2, _, _, _, _ = stats.linregress(edu_hat_w, log_income)
    f_stat = (s1 ** 2) / (1 / n)
    weak_results.append((z_strength, s2, f_stat))
    print(f"  Z强度={z_strength:.1f}  →  IV估计={s2:.3f}  F={f_stat:.1f}  {'⚠ 弱工具变量!' if f_stat < 10 else '✓'}")

# ============================================================
# 6. 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) OLS vs IV
axes[0].scatter(education, log_income, alpha=0.1, s=5, c=U, cmap="RdYlBu")
x_line = np.linspace(education.min(), education.max(), 100)
axes[0].plot(x_line, intercept_ols + slope_ols * x_line, "r-", linewidth=2,
             label=f"OLS: β={slope_ols:.3f} (biased)")
axes[0].plot(x_line, intercept_iv + slope_iv * x_line, "g--", linewidth=2,
             label=f"IV:  β={slope_iv:.3f} (correct)")
axes[0].set_xlabel("Education (years)")
axes[0].set_ylabel("Log Income")
axes[0].set_title("OLS vs IV Estimation")
axes[0].legend()

# (b) 第一阶段: Z → Education
axes[1].boxplot([df[df.Z == 0]["education"], df[df.Z == 1]["education"]],
                labels=["Z=0 (下半年)", "Z=1 (上半年)"])
axes[1].set_ylabel("Education (years)")
axes[1].set_title(f"First Stage: Z → Education\nCoeff = {slope_1st:.2f}")

# (c) DAG 示意
axes[2].set_xlim(0, 10)
axes[2].set_ylim(0, 10)
axes[2].text(1, 5, "Z\n(出生季度)", ha="center", va="center",
             bbox=dict(boxstyle="round", facecolor="lightblue"), fontsize=12)
axes[2].text(5, 5, "T\n(教育)", ha="center", va="center",
             bbox=dict(boxstyle="round", facecolor="lightyellow"), fontsize=12)
axes[2].text(9, 5, "Y\n(收入)", ha="center", va="center",
             bbox=dict(boxstyle="round", facecolor="lightgreen"), fontsize=12)
axes[2].text(7, 8.5, "U\n(天赋)", ha="center", va="center",
             bbox=dict(boxstyle="round", facecolor="lightsalmon"), fontsize=12)
axes[2].annotate("", xy=(3.5, 5), xytext=(2.5, 5),
                 arrowprops=dict(arrowstyle="->", lw=2, color="blue"))
axes[2].annotate("", xy=(7.5, 5), xytext=(6.5, 5),
                 arrowprops=dict(arrowstyle="->", lw=2, color="green"))
axes[2].annotate("", xy=(5.5, 6.5), xytext=(6.5, 8),
                 arrowprops=dict(arrowstyle="->", lw=2, color="red"))
axes[2].annotate("", xy=(8.5, 6.5), xytext=(7.5, 8),
                 arrowprops=dict(arrowstyle="->", lw=2, color="red"))
axes[2].set_title("IV DAG: Z → T → Y, U → T & Y")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("output/03_instrumental_variables.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n图表已保存: output/03_instrumental_variables.png")

print(f"""
{'=' * 60}
总结:
  真实 β:    {TRUE_BETA:.4f}
  OLS (有偏): {slope_ols:.4f}  ← 天赋混杂导致向上偏
  IV/2SLS:   {slope_iv:.4f}  ← 工具变量消除偏差
  Wald:      {wald:.4f}  ← 简化版IV

工具变量三条件:
  1. 相关性: Z 与 T 相关 (第一阶段 F > 10)
  2. 排他性: Z 只通过 T 影响 Y
  3. 外生性: Z 与 U 独立
{'=' * 60}
""")
