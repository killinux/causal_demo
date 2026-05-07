# 因果推断 (Causal Inference) 学习 Demo

## 一、什么是因果推断？

因果推断的核心问题：**"X 是否导致了 Y？"**

与传统统计学关注"相关性"不同，因果推断试图回答"因果性"问题。

> "Correlation is not causation" — 相关不等于因果

### 核心概念

| 概念 | 解释 | 例子 |
|------|------|------|
| **Treatment (处理)** | 我们想研究其效果的干预 | 吃药 vs 不吃药 |
| **Outcome (结果)** | 我们关心的结果变量 | 是否康复 |
| **Confounder (混杂因子)** | 同时影响处理和结果的变量 | 年龄（年龄影响是否吃药，也影响康复） |
| **Counterfactual (反事实)** | "如果当初没吃药，会怎样？" | 无法同时观测到的平行世界 |
| **ATE (平均处理效应)** | E[Y(1)] - E[Y(0)]，处理组与对照组的平均差异 | 吃药平均多恢复几天 |

### 两大理论框架

1. **Rubin 潜在结果框架 (Potential Outcomes)**：每个个体有两个潜在结果 Y(1) 和 Y(0)，因果效应 = Y(1) - Y(0)
2. **Pearl 结构因果模型 (SCM)**：用 DAG（有向无环图）表示变量间的因果关系

---

## 二、推荐入门论文与书籍

### 必读论文

1. **Rubin (1974)** - "Estimating Causal Effects of Treatments in Randomized and Nonrandomized Studies"
   - 潜在结果框架的奠基之作
   - 提出了"Rubin 因果模型"

2. **Rosenbaum & Rubin (1983)** - "The Central Role of the Propensity Score in Observational Studies for Causal Effects"
   - 倾向得分匹配的开创性论文
   - Demo: `02_propensity_score.py`

3. **Angrist & Imbens (1994)** - "Identification and Estimation of Local Average Treatment Effects"
   - 工具变量方法，获2021年诺贝尔经济学奖
   - Demo: `03_instrumental_variables.py`

4. **Pearl (1995)** - "Causal Diagrams for Empirical Research"
   - 因果图（DAG）的里程碑论文
   - Demo: `06_dag_and_do_calculus.py`

5. **Pearl (2009)** - "Causal Inference in Statistics: An Overview"
   - 综述性文章，适合入门

### 经典方法论文

6. **Card & Krueger (1994)** - "Minimum Wages and Employment"
   - 双重差分法（DID）经典案例
   - Demo: `04_difference_in_differences.py`

7. **Thistlethwaite & Campbell (1960)** / **Imbens & Lemieux (2008)**
   - 断点回归（RDD）
   - Demo: `05_regression_discontinuity.py`

8. **Abadie (2003)** / **Abadie, Diamond & Hainmueller (2010)**
   - 合成控制法 (Synthetic Control)

### 推荐书籍

- **《The Book of Why》** - Judea Pearl（科普向，中文版《为什么》）
- **"Causality" (2nd ed.)** - Judea Pearl（理论向）
- **"Causal Inference: The Mixtape"** - Scott Cunningham（实践向，免费在线）
- **"The Effect"** - Nick Huntington-Klein（入门友好，免费在线）
- **《基本无害的计量经济学》** - Angrist & Pischke

---

## 三、Demo 列表

| 文件 | 方法 | 核心思想 |
|------|------|----------|
| `01_ate_simulation.py` | ATE 估计 + Simpson's Paradox | 为什么简单比较均值会得到错误结论 |
| `02_propensity_score.py` | 倾向得分匹配 (PSM) | 用倾向得分消除选择偏差 |
| `03_instrumental_variables.py` | 工具变量 (IV) | 用外生变量解决内生性问题 |
| `04_difference_in_differences.py` | 双重差分 (DID) | 用时间维度消除不可观测混杂 |
| `05_regression_discontinuity.py` | 断点回归 (RDD) | 利用政策阈值的"准实验" |
| `06_dag_and_do_calculus.py` | DAG + do-演算 | Pearl 因果图方法 |
| `07_dowhy_demo.py` | DoWhy 框架 | 端到端因果推断流程 |

### 运行方式

```bash
pip install -r requirements.txt
python 01_ate_simulation.py
python 02_propensity_score.py
# ... 以此类推
```

每个 demo 都会生成可视化图表到 `output/` 目录。

---

## 四、学习路线建议

```
第1周: 理解问题 → 01_ate (Simpson悖论) → 读 Pearl (2009) 综述
第2周: PSM → 02_propensity_score → 读 Rosenbaum & Rubin (1983)
第3周: IV → 03_instrumental_variables → 读 Angrist & Imbens (1994)
第4周: DID → 04_did → 读 Card & Krueger (1994)
第5周: RDD → 05_rdd → 读 Imbens & Lemieux (2008)
第6周: DAG → 06_dag → 读 Pearl (1995) + 《The Book of Why》
第7周: 综合 → 07_dowhy → 尝试用真实数据做因果分析
```
