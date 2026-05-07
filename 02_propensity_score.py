"""
Demo 2: 倾向得分匹配 (Propensity Score Matching, PSM)
======================================================
论文: Rosenbaum & Rubin (1983)

核心思想:
  观测数据中, 处理组和对照组的协变量分布不同 (选择偏差).
  倾向得分 e(X) = P(T=1|X) 将高维协变量压缩为一维,
  匹配倾向得分相近的个体, 模拟"准随机实验".
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import os

np.random.seed(42)
os.makedirs("output", exist_ok=True)

n = 2000
TRUE_ATE = 5.0

# 协变量
age = np.random.normal(50, 10, n)
income = np.random.normal(50000, 15000, n)
education = np.random.randint(1, 5, n)

# 处理分配 (受协变量影响 — 非随机)
logit = -2 + 0.04 * age + 0.00003 * income + 0.5 * education
prob_t = 1 / (1 + np.exp(-logit))
treatment = np.random.binomial(1, prob_t)

# 结果 (真实效应 = 5)
outcome = 20 + 0.3 * age - 0.0001 * income + 2 * education + TRUE_ATE * treatment + np.random.normal(0, 3, n)

df = pd.DataFrame({
    "age": age, "income": income, "education": education,
    "treatment": treatment, "outcome": outcome
})

print("=" * 60)
print("Demo 2: 倾向得分匹配 (PSM)")
print("=" * 60)
print(f"真实 ATE: {TRUE_ATE}")

# ============================================================
# 1. Naive 估计
# ============================================================
naive = df[df.treatment == 1]["outcome"].mean() - df[df.treatment == 0]["outcome"].mean()
print(f"\nNaive ATE: {naive:.2f} (有偏)")

# ============================================================
# 2. 估计倾向得分
# ============================================================
X = df[["age", "income", "education"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
lr = LogisticRegression(max_iter=1000)
lr.fit(X_scaled, treatment)
df["pscore"] = lr.predict_proba(X_scaled)[:, 1]

# ============================================================
# 3. 最近邻匹配
# ============================================================
treated = df[df.treatment == 1].copy()
control = df[df.treatment == 0].copy()

nn = NearestNeighbors(n_neighbors=1)
nn.fit(control[["pscore"]])
distances, indices = nn.kneighbors(treated[["pscore"]])

matched_control = control.iloc[indices.flatten()]
psm_ate = treated["outcome"].values.mean() - matched_control["outcome"].values.mean()
print(f"PSM ATE:   {psm_ate:.2f} (匹配后)")

# ============================================================
# 4. IPW (逆概率加权) 估计
# ============================================================
df["weight"] = np.where(
    df.treatment == 1,
    1 / df.pscore,
    1 / (1 - df.pscore)
)
ipw_ate = (
    (df[df.treatment == 1]["outcome"] * df[df.treatment == 1]["weight"]).sum() /
    df[df.treatment == 1]["weight"].sum() -
    (df[df.treatment == 0]["outcome"] * df[df.treatment == 0]["weight"]).sum() /
    df[df.treatment == 0]["weight"].sum()
)
print(f"IPW ATE:   {ipw_ate:.2f} (逆概率加权)")

# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) 匹配前: 倾向得分分布
axes[0].hist(df[df.treatment == 0]["pscore"], bins=30, alpha=0.6, label="Control", density=True)
axes[0].hist(df[df.treatment == 1]["pscore"], bins=30, alpha=0.6, label="Treated", density=True)
axes[0].set_title("Before Matching: Propensity Score Distribution")
axes[0].set_xlabel("Propensity Score")
axes[0].legend()

# (b) 匹配后: 倾向得分分布
axes[1].hist(matched_control["pscore"].values, bins=30, alpha=0.6, label="Matched Control", density=True)
axes[1].hist(treated["pscore"].values, bins=30, alpha=0.6, label="Treated", density=True)
axes[1].set_title("After Matching: Propensity Score Distribution")
axes[1].set_xlabel("Propensity Score")
axes[1].legend()

# (c) 协变量平衡检查
covariates = ["age", "income", "education"]
before_smd = []
after_smd = []
for col in covariates:
    smd_before = abs(treated[col].mean() - control[col].mean()) / control[col].std()
    smd_after = abs(treated[col].mean() - matched_control[col].values.mean()) / matched_control[col].values.std()
    before_smd.append(smd_before)
    after_smd.append(smd_after)

x_pos = np.arange(len(covariates))
axes[2].barh(x_pos - 0.15, before_smd, height=0.3, label="Before Matching", color="red", alpha=0.7)
axes[2].barh(x_pos + 0.15, after_smd, height=0.3, label="After Matching", color="green", alpha=0.7)
axes[2].axvline(0.1, color="black", linestyle="--", label="SMD = 0.1 threshold")
axes[2].set_yticks(x_pos)
axes[2].set_yticklabels(covariates)
axes[2].set_xlabel("Standardized Mean Difference")
axes[2].set_title("Covariate Balance Check")
axes[2].legend()

plt.tight_layout()
plt.savefig("output/02_propensity_score.png", dpi=150, bbox_inches="tight")
plt.close()
print("图表已保存: output/02_propensity_score.png")

print(f"""
{'=' * 60}
总结:
  真实 ATE:  {TRUE_ATE:.2f}
  Naive:     {naive:.2f}   ← 有选择偏差
  PSM:       {psm_ate:.2f}   ← 匹配后消除偏差
  IPW:       {ipw_ate:.2f}   ← 逆概率加权也有效

关键假设:
  1. 无未观测混杂 (Unconfoundedness): 给定X, 处理分配与潜在结果独立
  2. 正值性 (Positivity): 0 < e(X) < 1, 每个人都有可能被处理
  3. SUTVA: 个体间无干扰
{'=' * 60}
""")
