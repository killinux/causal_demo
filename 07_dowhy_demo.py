"""
Demo 7: DoWhy 端到端因果推断框架
=================================
DoWhy 提供了一套完整的因果推断流程:
  1. Model   — 建立因果图 (DAG)
  2. Identify — 自动判断因果效应是否可识别, 找到估计策略
  3. Estimate — 用多种方法估计因果效应
  4. Refute  — 反驳检验, 验证结果的稳健性
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
os.makedirs("output", exist_ok=True)

print("=" * 60)
print("Demo 7: DoWhy 端到端因果推断")
print("=" * 60)

# ============================================================
# 1. 生成数据: 在线广告对购买的因果效应
# ============================================================
n = 5000
TRUE_ATE = 0.3

age = np.random.normal(35, 10, n)
income = np.random.normal(50, 15, n)
web_activity = np.random.normal(5, 2, n)

# 广告投放 (受用户特征影响 — 非随机)
logit = -1 + 0.02 * age + 0.01 * income + 0.2 * web_activity
prob_ad = 1 / (1 + np.exp(-logit))
ad_shown = np.random.binomial(1, prob_ad)

# 购买 (受广告和用户特征影响)
prob_buy = 1 / (1 + np.exp(-(
    -3 + 0.03 * age + 0.02 * income + 0.1 * web_activity + TRUE_ATE * ad_shown
)))
purchase = np.random.binomial(1, prob_buy)

df = pd.DataFrame({
    "age": age, "income": income, "web_activity": web_activity,
    "ad_shown": ad_shown, "purchase": purchase
})

print(f"场景: 在线广告是否导致用户购买?")
print(f"真实 ATE: {TRUE_ATE}")
print(f"样本量: {n}, 广告率: {ad_shown.mean():.2%}, 购买率: {purchase.mean():.2%}")

# ============================================================
# 2. DoWhy 四步流程
# ============================================================
try:
    import dowhy
    from dowhy import CausalModel

    # Step 1: 建立因果模型
    print(f"\n--- Step 1: Model (建立因果图) ---")
    model = CausalModel(
        data=df,
        treatment="ad_shown",
        outcome="purchase",
        common_causes=["age", "income", "web_activity"],
        graph=None  # 也可以传入 DOT 格式的 DAG
    )
    print("  因果图已建立")
    print("  处理变量: ad_shown")
    print("  结果变量: purchase")
    print("  混杂因子: age, income, web_activity")

    # Step 2: 识别因果效应
    print(f"\n--- Step 2: Identify (识别估计策略) ---")
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    print(f"  估计策略: {identified.estimands}")

    # Step 3: 估计因果效应
    print(f"\n--- Step 3: Estimate (多方法估计) ---")

    # 方法1: 倾向得分匹配
    estimate_psm = model.estimate_effect(
        identified,
        method_name="backdoor.propensity_score_matching"
    )
    print(f"  PSM:     ATE = {estimate_psm.value:.4f}")

    # 方法2: 逆概率加权
    estimate_ipw = model.estimate_effect(
        identified,
        method_name="backdoor.propensity_score_weighting"
    )
    print(f"  IPW:     ATE = {estimate_ipw.value:.4f}")

    # 方法3: 线性回归
    estimate_lr = model.estimate_effect(
        identified,
        method_name="backdoor.linear_regression"
    )
    print(f"  线性回归: ATE = {estimate_lr.value:.4f}")

    # Step 4: 反驳检验
    print(f"\n--- Step 4: Refute (稳健性检验) ---")

    # 安慰剂检验 (随机替换处理变量)
    refute_placebo = model.refute_estimate(
        identified, estimate_psm,
        method_name="placebo_treatment_refuter",
        placebo_type="permute"
    )
    print(f"  安慰剂检验: ATE = {refute_placebo.new_effect:.4f} (应接近0)")

    # 子集检验 (随机子集)
    refute_subset = model.refute_estimate(
        identified, estimate_psm,
        method_name="data_subset_refuter",
        subset_fraction=0.8
    )
    print(f"  子集检验:   ATE = {refute_subset.new_effect:.4f} (应与原估计接近)")

    print(f"""
{'=' * 60}
DoWhy 四步总结:
  1. Model:    定义 DAG, 明确处理/结果/混杂变量
  2. Identify: 自动推导估计策略 (后门/前门/IV)
  3. Estimate: 多方法估计, 交叉验证
  4. Refute:   安慰剂/子集/随机混杂 — 检验稳健性

真实 ATE = {TRUE_ATE}
PSM = {estimate_psm.value:.4f}
IPW = {estimate_ipw.value:.4f}
LR  = {estimate_lr.value:.4f}
{'=' * 60}
""")

except ImportError:
    print("\nDoWhy 未安装, 使用手动实现演示同样的流程...")

    # 手动实现后门调整
    from sklearn.linear_model import LogisticRegression, LinearRegression

    # IPW
    X = df[["age", "income", "web_activity"]]
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X, df["ad_shown"])
    pscore = lr.predict_proba(X)[:, 1]
    pscore = np.clip(pscore, 0.05, 0.95)

    weights = np.where(df["ad_shown"] == 1, 1 / pscore, 1 / (1 - pscore))
    ipw_ate = (
        np.average(df[df.ad_shown == 1]["purchase"], weights=weights[df.ad_shown == 1]) -
        np.average(df[df.ad_shown == 0]["purchase"], weights=weights[df.ad_shown == 0])
    )
    print(f"  IPW ATE: {ipw_ate:.4f}")

    # 线性回归
    reg = LinearRegression()
    reg.fit(df[["ad_shown", "age", "income", "web_activity"]], df["purchase"])
    lr_ate = reg.coef_[0]
    print(f"  LR ATE:  {lr_ate:.4f}")

    # 安慰剂检验
    df_placebo = df.copy()
    df_placebo["ad_shown"] = np.random.permutation(df_placebo["ad_shown"].values)
    reg_p = LinearRegression()
    reg_p.fit(df_placebo[["ad_shown", "age", "income", "web_activity"]], df_placebo["purchase"])
    placebo_ate = reg_p.coef_[0]
    print(f"  安慰剂:  {placebo_ate:.4f} (应接近0)")

    print(f"""
{'=' * 60}
手动实现总结 (DoWhy未安装时的后备方案):
  真实 ATE = {TRUE_ATE}
  IPW ATE  = {ipw_ate:.4f}
  LR ATE   = {lr_ate:.4f}
  安慰剂   = {placebo_ate:.4f} (确认非虚假效应)

安装 DoWhy: pip install dowhy
然后重新运行可获得完整的四步流程.
{'=' * 60}
""")
