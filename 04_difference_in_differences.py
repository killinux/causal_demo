"""
Demo 4: 双重差分法 (Difference-in-Differences, DID)
====================================================
论文: Card & Krueger (1994)

核心思想:
  DID = (Y_treated_after - Y_treated_before) - (Y_control_after - Y_control_before)
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

TRUE_DID = 2.5
n_stores = 100
n_periods = 8

records = []
for store in range(n_stores):
    treated = store < n_stores // 2
    store_fe = np.random.normal(0, 3)
    for t in range(n_periods):
        trend = 0.5 * t
        post = t >= 4
        effect = TRUE_DID * treated * post
        employment = 20 + store_fe + trend + 3 * treated + effect + np.random.normal(0, 1)
        records.append({
            "store": store, "time": t,
            "treated": int(treated), "post": int(post),
            "employment": employment
        })

df = pd.DataFrame(records)

print("=" * 60)
print("Demo 4: DID")
print("=" * 60)
print(f"真实政策效应: {TRUE_DID}")

# 四格均值 DID
means = df.groupby(["treated", "post"])["employment"].mean().unstack()
print(f"\n四格均值表:")
print(f"              干预前    干预后    差值")
print(f"  处理组:     {means.loc[1, 0]:.2f}    {means.loc[1, 1]:.2f}    {means.loc[1, 1] - means.loc[1, 0]:.2f}")
print(f"  对照组:     {means.loc[0, 0]:.2f}    {means.loc[0, 1]:.2f}    {means.loc[0, 1] - means.loc[0, 0]:.2f}")

did_manual = (means.loc[1, 1] - means.loc[1, 0]) - (means.loc[0, 1] - means.loc[0, 0])
print(f"\n手动 DID = {did_manual:.2f}")

# 回归 DID
df["treat_post"] = df["treated"] * df["post"]
reg = LinearRegression()
reg.fit(df[["treated", "post", "treat_post"]], df["employment"])
did_reg = reg.coef_[2]
print(f"\n回归 DID: β = {did_reg:.4f}")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

treated_means = df[df.treated == 1].groupby("time")["employment"].mean()
control_means = df[df.treated == 0].groupby("time")["employment"].mean()

axes[0].plot(treated_means.index, treated_means.values, "ro-", label="Treated (NJ)", linewidth=2)
axes[0].plot(control_means.index, control_means.values, "bs-", label="Control (PA)", linewidth=2)
axes[0].axvline(3.5, color="gray", linestyle="--", label="Policy Change")
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Employment")
axes[0].set_title("DID: Time Trends")
axes[0].legend()

# DID 几何解释
y_ctrl_pre = means.loc[0, 0]
y_ctrl_post = means.loc[0, 1]
y_treat_pre = means.loc[1, 0]
y_treat_post = means.loc[1, 1]
y_counterfactual = y_treat_pre + (y_ctrl_post - y_ctrl_pre)

axes[1].set_xlim(-0.5, 1.5)
axes[1].plot([0, 1], [y_ctrl_pre, y_ctrl_post], "bs-", linewidth=2, markersize=10, label="Control")
axes[1].plot([0, 1], [y_treat_pre, y_treat_post], "ro-", linewidth=2, markersize=10, label="Treated (actual)")
axes[1].plot([0, 1], [y_treat_pre, y_counterfactual], "r--", linewidth=2, alpha=0.5, label="Counterfactual")
axes[1].annotate("", xy=(1.1, y_treat_post), xytext=(1.1, y_counterfactual),
                 arrowprops=dict(arrowstyle="<->", color="green", lw=2))
axes[1].text(1.15, (y_treat_post + y_counterfactual) / 2, f"DID={did_manual:.1f}",
             fontsize=12, color="green", fontweight="bold")
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(["Pre", "Post"])
axes[1].set_ylabel("Employment")
axes[1].set_title("DID Geometry")
axes[1].legend()

plt.tight_layout()
plt.savefig("output/04_did.png", dpi=150, bbox_inches="tight")
plt.close()
print("图表已保存: output/04_did.png")

print(f"""
{'=' * 60}
总结:
  真实效应:   {TRUE_DID:.2f}
  手动 DID:   {did_manual:.2f}
  回归 DID:   {did_reg:.2f}

关键假设: 平行趋势
  如果没有干预, 处理组和对照组会有相同的时间趋势.
{'=' * 60}
""")
