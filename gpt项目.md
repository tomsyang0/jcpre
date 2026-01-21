## 预测文件详细说明

### 文件位置
`outputs/jc_fbref_future_20260120_143944_pred.csv`

### 核心预测列说明

#### 1. 基础信息列
- `match_id`, `match_num`: 比赛唯一标识
- `jc_date`: 比赛日期
- `league_name_jc`: 联赛名称
- `home_team_cn`, `away_team_cn`: 主客队中文名
- `home_team_fbref`, `away_team_fbref`: 主客队FBref英文名（用于特征匹配）

#### 2. 市场赔率与概率
- `spf_sp_home_now`, `spf_sp_draw_now`, `spf_sp_away_now`: 当前胜平负赔率
- `spf_p_home_now`, `spf_p_draw_now`, `spf_p_away_now`: 当前胜平负隐含概率
- `hcp_sp_home_now`, `hcp_sp_draw_now`, `hcp_sp_away_now`: 当前让球胜平负赔率
- `hcp_p_home_now`, `hcp_p_draw_now`, `hcp_p_away_now`: 当前让球胜平负隐含概率

#### 3. 模型预测概率（核心输出）
- `pred_prob_home`, `pred_prob_draw`, `pred_prob_away`: **模型预测的胜平负概率**
  - 基于机器学习、Poisson、Elo、市场概率的融合结果
  - 使用动态权重调整，考虑数据完整性
- `pred_hhad_prob_home`, `pred_hhad_prob_draw`, `pred_hhad_prob_away`: **模型预测的让球胜平负概率**
  - 机器学习模型与Poisson模型的加权平均

#### 4. 期望价值（EV）与凯利投注
- `ev_home`, `ev_draw`, `ev_away`: **胜平负期望价值**
  - 正值表示有投注价值，负值表示没有价值
  - 计算公式：EV = (预测概率 × 赔率) - 1
- `kelly_home`, `kelly_draw`, `kelly_away`: **凯利投注比例**
  - 建议投注资金比例（0-1之间）
  - 0表示不投注，0.05表示建议投注5%的资金
- `hhad_ev_home`, `hhad_ev_draw`, `hhad_ev_away`: 让球胜平负期望价值
- `hhad_kelly_home`, `hhad_kelly_draw`, `hhad_kelly_away`: 让球胜平负凯利比例

#### 5. 最佳投注推荐
- `best_bet_market`: 推荐投注市场（SPF_HOME/SPF_DRAW/SPF_AWAY/HHAD_HOME等）
- `best_bet_ev`: 推荐投注的期望价值
- `best_bet_kelly_stake`: 推荐投注的凯利比例

#### 6. 其他预测信息
- `pred_lambda_home`, `pred_lambda_away`: Poisson模型预测的进球期望值
- `pred_prob_over25`, `pred_prob_under25`: 大于2.5球/小于2.5球概率
- `pred_prob_btts_yes`: 双方都进球概率
- `pred_score_mode`: 最可能比分（如"1-0"）
- `pred_score_top3`: 前三可能比分及概率（如"1-0:0.132|1-1:0.111|0-1:0.109"）

#### 7. 市场对比与边缘
- `mkt_prob_home`, `mkt_prob_draw`, `mkt_prob_away`: 市场隐含概率
- `edge_home`, `edge_draw`, `edge_away`: **模型概率与市场概率的差值**
  - 正值表示模型认为市场低估了该结果
  - 负值表示模型认为市场高估了该结果

#### 8. 特征数据
- `home_GF`, `home_GA`, `home_xG`, `home_xGA`等: 球队FBref统计数据
- `home_def_strength`, `home_off_buildup`, `home_tempo`: PCA提取的球队特征因子
- `tempo_diff`: 主客队节奏差异

#### 9. 支持率数据
- `had_support_h`, `had_support_d`, `had_support_a`: 胜平负支持率
- `had_prob_h`, `had_prob_d`, `had_prob_a`: 胜平负概率（平台计算）
- `hhad_support_h`, `hhad_support_d`, `hhad_support_a`: 让球胜平负支持率

### 使用指南

#### 1. 筛选有价值投注
```python
import pandas as pd
df = pd.read_csv("outputs/jc_fbref_future_20260120_143944_pred.csv")

# 筛选有正期望价值的投注
value_bets = df[df["best_bet_ev"] > 0]

# 筛选边缘较大的比赛（模型与市场差异大）
edge_bets = df[df["edge_home"].abs() > 0.05 | df["edge_draw"].abs() > 0.05 | df["edge_away"].abs() > 0.05]
```

#### 2. 投注策略建议
- **保守策略**: 只投注`best_bet_ev > 0.1`且`best_bet_kelly_stake > 0.02`的比赛
- **激进策略**: 投注所有`best_bet_ev > 0`的比赛，按凯利比例分配资金
- **边缘策略**: 重点关注`edge_*`绝对值大的比赛，模型与市场分歧明显

#### 3. 风险控制
- 凯利比例`kelly_*`建议最大不超过0.05（5%资金）
- 避免投注`pred_prob_* < 0.2`的高赔率选项
- 注意比赛数据完整性：`home_team_fbref`为空表示缺少FBref数据，模型权重会打折

#### 4. 比分预测使用
- `pred_score_mode`: 用于比分投注参考
- `pred_score_top3`: 前三可能比分，可用于组合投注
- `pred_prob_over25`: 用于大小球投注决策

#### 5. 实时监控
- 比赛开始前可对比当前赔率与预测时的赔率变化
- 支持率变化（`had_support_*`）可作为市场情绪参考
- 赔率变动（`spf_sp_*_open` vs `spf_sp_*_close`）反映市场调整

### 示例分析
以第一场比赛为例（日本U23 vs 韩国U23）：
- 模型预测概率：主胜52.5%，平局28.1%，客胜19.4%
- 市场隐含概率：主胜53.7%，平局28.6%，客胜17.7%
- 边缘：主胜-1.18%（模型略低于市场），客胜+1.68%（模型高于市场）
- 最佳投注：`NO_BET`（无正期望价值投注）
- 最可能比分：1-0（概率13.2%）

### 注意事项
1. 预测基于历史数据和统计模型，不保证100%准确
2. 建议结合其他信息源（球队新闻、伤病等）综合判断
3. 资金管理比单场预测更重要，严格执行凯利比例
4. 定期回测和调整投注策略

预测文件提供了全面的量化分析，可作为投注决策的重要参考工具。




## 项目运行问题解决方案

### 问题分析
项目无法直接在根目录下运行的主要原因是：
1. **导入路径问题**：模块使用相对导入`from fp.xxx`，在根目录下运行时Python无法找到这些模块
2. **缺少运行脚本**：项目缺少直接运行的入口脚本

### 已完成的修复

#### 1. 修复导入路径
- 修改了`src/fp/train.py`、`src/fp/features.py`、`src/fp/predict.py`等文件中的导入语句
- 将`from fp.xxx`改为`from src.fp.xxx`，确保从项目根目录可以正确导入

#### 2. 创建运行脚本
- **训练脚本**：`run_train.py` - 已存在并可以正常运行
- **预测脚本**：`run_predict.py` - 已创建并测试成功
- **数据采集脚本**：`today_data.py` - 已存在并可以正常运行

#### 3. 修复校准文件问题
- 修复了`src/fp/calibration.py`中的`LGBMWrapper`类
- 添加了序列化支持和sklearn兼容性修复
- 训练脚本现在可以成功生成校准器文件

### 运行方法

#### 方法一：使用提供的运行脚本
```bash
# 1. 数据采集（获取最新比赛数据）
python today_data.py

# 2. 训练模型（如果还没有训练过）
python run_train.py

# 3. 进行预测
python run_predict.py
```

#### 方法二：直接运行模块
```bash
# 1. 训练模型
python -m src.fp.train --train_csv datasets/jc_fbref_bonus_support_2025-01-01_to_2026-01-17.csv --artifacts_dir artifacts

# 2. 进行预测
python -m src.fp.predict --future_glob "datasets/jc_fbref_future_*.csv" --artifacts_dir artifacts --out_dir outputs
```

#### 方法三：使用绝对导入
在项目根目录创建自己的脚本时，需要添加路径：
```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```

### 验证结果
1. ✅ `run_train.py` - 训练成功，生成校准器文件
2. ✅ `run_predict.py` - 预测成功，生成预测结果文件
3. ✅ `today_data.py` - 数据采集成功，生成未来比赛数据

### 文件结构说明
```
e:\jcpre\
├── run_train.py      # 训练入口
├── run_predict.py    # 预测入口（新增）
├── today_data.py     # 数据采集入口
├── artifacts/        # 模型和校准器文件
├── datasets/         # 数据文件
├── outputs/          # 预测输出
└── src/              # 源代码
```

现在项目可以在根目录下正常运行，所有主要功能都已修复并测试通过。



下面给你一个可落地的“足球预测 + 投注建议”项目方案（含目录结构、建模思路、训练/预测脚本设计、缺失值策略、输出字段规范、README 使用说明模板）。核心目标是：

1) **历史数据训练一次**并保存模型与预处理器（artifact），  
2) 之后对任意 `jc_fbref_future_*.csv` **直接加载 artifact 预测**，不重复学习，  
3) 同时给出 **胜平负/让球胜平负概率、比分分布、大小球、价值投注与凯利建议**。

> 说明：我无法直接读取你本地文件并运行，但下面的结构和代码骨架是按你的字段设计的，复制到工程里即可逐步跑通。

---

## 1. 项目总体方案（多模型 + 自适应融合）

### 1.1 数据特点与对应策略
- **fbref 覆盖不全**：fbref 指标缺失时不能丢弃比赛  
  → 采用“缺失可用”的特征工程（缺失指示器 + 分层回退）+ 一个不依赖 fbref 的模型（Elo/市场模型）。
- **竞彩赔率可能未开（空值）**：赔率特征缺失  
  → 市场模型不可用时，用“基本面模型”（Poisson/Elo/ML）顶上；融合时动态调整权重。
- **要预测比分**：需要从“进球分布”出发  
  → 用 **高级泊松（Dixon-Coles/双变量泊松）** 输出 `(λ_home, λ_away)`，再做 **蒙特卡罗模拟**得到比分矩阵与派生盘口概率。
- **要给投注建议**：需要“模型概率 vs 市场隐含概率”的差  
  → 计算去水后的市场隐含概率，做 **EV、Kelly、价值洼地**筛选。

### 1.2 模型组件（建议至少四类）
1) **Dixon–Coles（DC）时间衰减泊松模型（按联赛/全局回退）**  
   - 输入：历史比分（home_goals/away_goals）、球队ID、日期  
   - 输出：每场 `λ_home, λ_away` + 低比分相关修正  
   - 优点：比分可解释、可生成全比分分布

2) **Elo（按联赛）+ 主场优势 + 近期状态修正**  
   - 输入：历史赛果（胜平负）、日期  
   - 输出：胜平负概率基线 + 期望净胜球（可映射到 λ 近似）
   - 优点：不依赖 fbref/赔率，覆盖最全

3) **机器学习分类模型（胜平负 & 让球胜平负）**（LightGBM/CatBoost 二选一）  
   - 输入：  
     - fbref 指标（xG/xGA/tempo/def_strength/off_buildup 等）与差分  
     - 赔率与盘口（open/close/now，若缺失则缺失指示器）  
     - 市场热度/心理误差字段（had_support、had_psy_err 等，若未来文件里有就用）  
     - Elo/Poisson 输出作为“二级特征”  
   - 输出：校准后的 `P(H/D/A)`、`P(HHAD)`  
   - 优点：利用你手上“混合信息”最大化拟合能力

4) **市场模型（赔率隐含概率 + 去水 + 盘口一致性修正）**  
   - 输入：spf/hcp 当前赔率（或 close）  
   - 输出：市场隐含概率（作为强 baseline）  
   - 优点：赔率往往包含信息；缺点：赔率缺失时不能用

### 1.3 融合（关键：缺什么就用什么）
对每场比赛，根据可用信息动态融合：

- 若 **赔率齐全**：市场模型权重更高（例如 0.45），其余分摊  
- 若 **fbref 缺失**：ML 权重降低，Poisson/Elo 提升  
- 若 **赔率缺失**：市场权重为 0，主要靠 Poisson/Elo/ML（无赔率特征也可训练一套 ML_fundamental）

最终输出：
- `prob_1x2_*`、`prob_hhad_*`
- `lambda_home/lambda_away`
- MonteCarlo 得到：`P(Over2.5)`, `P(BTTS)`, TopN 比分概率
- 投注建议：EV、Kelly、推荐投注项

---

## 2. 目录结构（可直接照搬）

```
football_predictor/
  pyproject.toml (或 requirements.txt)
  README.md
  datasets/
    jc_fbref_bonus_support_2025-01-01_to_2026-01-17.csv
    jc_fbref_future_*.csv
  artifacts/
    preprocess.pkl
    model_lgbm_1x2.pkl
    model_lgbm_hhad.pkl
    model_dc_poisson.pkl
    model_elo.pkl
    ensemble_config.json
    feature_schema.json
    train_meta.json
  outputs/
    jc_fbref_future_XXXX_pred.csv
  src/
    fp/
      __init__.py
      config.py
      io.py
      clean.py
      features.py
      calibration.py
      models/
        elo.py
        dc_poisson.py
        market.py
        ml.py
        ensemble.py
      betting.py
      simulate.py
      train.py
      predict.py
```

---

## 3. 训练与预测的“分步骤”流程

### Step 1）数据清洗与统一键
- 统一球队键：优先 `*_team_fbref`，否则用 `*_team_cn`
- 解析比分：`full_score_raw` → `home_goals/away_goals`
- 丢弃无效：`bonus_isCancel==1` 或 `match_status` 非完赛（历史训练集要完赛）
- 日期排序，严格防止“未来信息泄露”（例如不要用 close 赔率去预测 open 时点）

### Step 2）特征工程（可复用）
1) **fbref 基本面差分**  
   - `xG_diff = home_xG - away_xG`
   - `xGA_diff = home_xGA - away_xGA`
   - `tempo_diff`（你已有）
   - `home_attack = home_xG / (home_GF+ε)` 等派生可选
2) **赔率/盘口特征**（缺失就 NaN + indicator）  
   - spf/hcp 的 open/close/now
   - 赔率变化：`close-open`, `now-close`
   - 盘口变化：`hcp_line_close - hcp_line_open`
3) **市场隐含概率（去水）**  
   - `p = (1/odds) / sum(1/odds)`
4) **Elo 特征**（训练时按时间滚动；预测时从 artifact 的最后状态读）  
   - `elo_home, elo_away, elo_diff`
5) **Poisson 先验输出作为特征**  
   - `lambda_home_dc, lambda_away_dc`

> 注意：为了“预测一次后可复用”，你需要把 **预处理器（imputer/encoder）** 和 **Elo 当前分数表** 一起保存到 `artifacts/`。

### Step 3）模型训练（一次性）
- 训练 DC 泊松：按联赛拟合（样本不足则全局）
- 训练 Elo：按联赛更新 K 值，可用 time-decay
- 训练 ML（1x2、hhad 两个分类器）：使用上述特征
- 校准：对 ML 输出做 isotonic 或 Platt（保证概率可用）
- 学融合权重：在验证集上最小化 logloss/Brier，保存到 `ensemble_config.json`

### Step 4）预测（加载 artifact，无需重训）
- 读取 `jc_fbref_future_*.csv`
- 同样做清洗、特征
- 逐模型输出概率/λ
- 动态融合
- 蒙特卡罗模拟比分分布（也可直接用泊松矩阵解析计算）
- 计算 EV/Kelly，给出投注建议
- 输出到 `outputs/` 或 `datasets/`（按你要求同位置也可以）

---

## 4. 输出文件建议字段（在 future 原表基础上追加）

至少追加这些列（建议）：

**概率类**
- `pred_prob_home, pred_prob_draw, pred_prob_away`
- `pred_prob_over25, pred_prob_under25`
- `pred_prob_btts_yes`
- `pred_prob_hhad_home, pred_prob_hhad_draw, pred_prob_hhad_away`（让球胜平负，基于 hcp_line_now）

**进球/比分类**
- `pred_lambda_home, pred_lambda_away`
- `pred_score_mode`（最可能比分，如 `1-0`）
- `pred_score_top3`（如 `1-0:0.12|1-1:0.10|2-1:0.09`）
- `pred_home_goals_mean, pred_away_goals_mean`

**市场对比 & 投注建议**
- `mkt_prob_home, mkt_prob_draw, mkt_prob_away`（用 spf_sp_*_now 计算，缺失则 NaN）
- `edge_home, edge_draw, edge_away`（pred - mkt）
- `ev_home, ev_draw, ev_away`（odds*pred-1）
- `kelly_home, kelly_draw, kelly_away`（含上限风控）
- `best_bet_market`（如 `SPF_HOME` / `SPF_DRAW` / `HHAD_AWAY` / `NO_BET`）
- `best_bet_ev, best_bet_kelly_stake`

---

## 5. 关键实现要点（避免踩坑）

### 5.1 缺失值策略（强制要求）
- 所有数值特征：`NaN` 保留 + `is_missing_*` 指示器  
- fbref 全缺的比赛：  
  - Poisson/Elo 仍能预测  
  - ML 模型要么用 LightGBM（原生支持 NaN），要么做 impute
- 赔率缺失：市场模型输出 NaN，融合权重自动归零

### 5.2 训练/验证切分
用时间切分（例如最后 20% 日期作为验证），不要随机切分，避免泄露。

### 5.3 赔率“使用时点”
未来表里有 `*_now`，就以 now 为主；历史训练建议也用同口径（例如 close 或 now），否则“训练特征与预测特征分布不一致”。

---

## 6. 代码骨架（核心脚本接口）

下面给的是“你按这个写就能跑”的接口设计（非最短代码，但稳定可维护）。

### 6.1 requirements.txt（示例）
```txt
pandas>=2.1
numpy>=1.26
scipy>=1.11
scikit-learn>=1.4
lightgbm>=4.0
joblib>=1.3
tqdm>=4.66
python-dateutil>=2.9
```

### 6.2 命令行约定
- 训练：
```bash
python -m fp.train \
  --train_csv datasets/jc_fbref_bonus_support_2025-01-01_to_2026-01-17.csv \
  --artifacts_dir artifacts
```

- 预测（支持 glob）：
```bash
python -m fp.predict \
  --future_glob "datasets/jc_fbref_future_*.csv" \
  --artifacts_dir artifacts \
  --out_dir datasets
```

### 6.3 训练脚本要保存的内容（artifact）
- `preprocess.pkl`（特征列表、缺失处理、类别编码等）
- `model_lgbm_1x2.pkl`
- `model_lgbm_hhad.pkl`
- `model_dc_poisson.pkl`（各联赛参数/球队 attack/defense）
- `model_elo.pkl`（各联赛当前 elo 表 + 超参）
- `ensemble_config.json`（融合权重、动态规则）
- `feature_schema.json`（可追溯）
- `train_meta.json`（训练时间、样本量、版本）

---

## 7. README 使用说明文件（可直接放 README.md）

你可以直接复制下面这段作为 `README.md`：

```md
# Football Predictor (JC + FBRef Hybrid)

## 1. 功能
- 读取历史训练数据：jc_fbref_bonus_support_*.csv
- 训练并保存可复用模型（Poisson/DC、Elo、LightGBM、市场隐含概率、融合器）
- 读取待预测数据：jc_fbref_future_*.csv
- 输出：胜平负/让球胜平负概率、比分分布、大小球概率、EV/Kelly 投注建议

## 2. 数据位置
- 训练集：datasets/jc_fbref_bonus_support_2025-01-01_to_2026-01-17.csv
- 预测集：datasets/jc_fbref_future_*.csv
- 输出：datasets/*_pred.csv（或 outputs/）

## 3. 安装
```bash
pip install -r requirements.txt
```

## 4. 训练（只需一次）
```bash
python -m fp.train \
  --train_csv datasets/jc_fbref_bonus_support_2025-01-01_to_2026-01-17.csv \
  --artifacts_dir artifacts
```

训练完成后 artifacts/ 将包含：
- preprocess.pkl
- model_lgbm_1x2.pkl, model_lgbm_hhad.pkl
- model_dc_poisson.pkl
- model_elo.pkl
- ensemble_config.json

## 5. 预测（复用 artifacts，无需重训）
```bash
python -m fp.predict \
  --future_glob "datasets/jc_fbref_future_*.csv" \
  --artifacts_dir artifacts \
  --out_dir datasets
```

输出会在 datasets/ 下生成：
- jc_fbref_future_XXXX_pred.csv

## 6. 输出字段说明（节选）
- pred_prob_home/draw/away：模型融合后的胜平负概率
- pred_lambda_home/away：期望进球
- pred_score_mode：最可能比分
- mkt_prob_*：根据 spf_sp_*_now 计算的去水市场概率（若赔率缺失则为空）
- ev_*：期望收益（odds*prob-1）
- kelly_*：凯利下注比例（含风控上限）
- best_bet_market：推荐投注项（若无价值则 NO_BET）

## 7. 风险提示
本项目仅做数据分析与概率预测，不保证盈利。请做好资金管理与风险控制。
```

---

## 8. 你接下来需要我继续“分步骤落地”的方式

你已经给了字段和文件名。为了我下一步把**具体代码实现（可直接运行）**按模块逐个写出来，我需要你确认两个关键点（否则容易写偏）：

1) 历史训练集里用于训练的“赛果列”到底以哪组为准：  
   - `home_goals/away_goals` 是否总是可用且正确？  
   - 还是需要从 `full_score_raw` 解析为主？

2) 你希望预测输出文件放在：  
   - 仍在 `datasets/`（与你要求一致），还是 `outputs/`？

你回复这两点后，我会按“Step1～StepN”把 `src/fp/` 下每个文件的代码（清洗、特征、DC泊松、Elo、LightGBM、融合、模拟、投注建议、train/predict 脚本）逐个给出，并保证：
- 第一次 `train` 生成 artifacts  
- 之后任意 future 文件只跑 `predict` 即可复用模型与 Elo 状态。


下面按你要的目录，把**每个脚本的具体内容**（可直接复制落地）给出。该版本实现了：

- 历史数据训练一次，产出 `artifacts/`（可复用）
- 预测时加载 artifact，不重复训练
- 多模型：**市场隐含概率、Elo、（类）Dixon-Coles 攻防泊松、LightGBM** + 动态融合
- 蒙特卡罗/解析比分矩阵、大小球、让球、EV/Kelly、价值投注建议输出

> 说明：为了保证可运行与可维护，我实现的是“攻防泊松 + 低比分相关修正（ρ）”的 Dixon-Coles 核心思想（而不是最复杂的论文全量版本）。你后续可继续增强（如按联赛单独拟合、时间衰减更精细、分层贝叶斯等）。

---

## 0) 文件结构

```
football_predictor/
  requirements.txt
  README.md
  datasets/
    jc_fbref_bonus_support_2025-01-01_to_2026-01-17.csv
    jc_fbref_future_*.csv
  artifacts/
  src/
    fp/
      __init__.py
      config.py
      io.py
      clean.py
      features.py
      calibration.py
      simulate.py
      betting.py
      train.py
      predict.py
      models/
        __init__.py
        market.py
        elo.py
        dc_poisson.py
        ml.py
        ensemble.py
```

---

## 1) requirements.txt

```txt
pandas>=2.1
numpy>=1.26
scipy>=1.11
scikit-learn>=1.4
lightgbm>=4.0
joblib>=1.3
tqdm>=4.66
python-dateutil>=2.9
```

---

## 2) src/fp/__init__.py

```python
__all__ = ["config"]
__version__ = "0.1.0"
```

---

## 3) src/fp/config.py

```python
from __future__ import annotations

ARTIFACTS = {
    "preprocess": "preprocess.pkl",
    "model_ml_1x2": "model_lgbm_1x2.pkl",
    "model_ml_hhad": "model_lgbm_hhad.pkl",
    "calib_1x2": "calibrator_1x2.pkl",
    "calib_hhad": "calibrator_hhad.pkl",
    "model_elo": "model_elo.pkl",
    "model_poisson": "model_dc_poisson.pkl",
    "ensemble_cfg": "ensemble_config.json",
    "feature_schema": "feature_schema.json",
    "train_meta": "train_meta.json",
}

# 结果标签映射：0=主胜 1=平 2=客胜
CLASSES_1X2 = ["H", "D", "A"]

DATE_COL = "jc_date"

# 赔率列优先级：优先 now，其次 close，其次 open（训练集可能没有 now）
ODDS_SPF_CANDIDATES = [
    ("spf_sp_home_now", "spf_sp_draw_now", "spf_sp_away_now"),
    ("spf_sp_home_close", "spf_sp_draw_close", "spf_sp_away_close"),
    ("spf_sp_home_open", "spf_sp_draw_open", "spf_sp_away_open"),
]
ODDS_HCP_CANDIDATES = [
    ("hcp_sp_home_now", "hcp_sp_draw_now", "hcp_sp_away_now"),
    ("hcp_sp_home_close", "hcp_sp_draw_close", "hcp_sp_away_close"),
    ("hcp_sp_home_open", "hcp_sp_draw_open", "hcp_sp_away_open"),
]

HCP_LINE_CANDIDATES = ["hcp_line_now", "hcp_line_close", "hcp_line_open"]

# 训练时要过滤掉的无效比赛（若列存在）
CANCEL_COL = "bonus_isCancel"

# 输出比分矩阵上限
MAX_GOALS = 7  # 0..7
```

---

## 4) src/fp/io.py

```python
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
import pandas as pd


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def write_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def glob_paths(pattern: str) -> list[str]:
    return sorted(glob.glob(pattern))


@dataclass
class Paths:
    artifacts_dir: str
    out_dir: str | None = None

    def a(self, name: str) -> str:
        os.makedirs(self.artifacts_dir, exist_ok=True)
        return os.path.join(self.artifacts_dir, name)

    def o(self, name: str) -> str:
        assert self.out_dir is not None
        os.makedirs(self.out_dir, exist_ok=True)
        return os.path.join(self.out_dir, name)
```

---

## 5) src/fp/clean.py

```python
from __future__ import annotations

import numpy as np
import pandas as pd


def _parse_score_str(s: str):
    # 支持 "1:0" / "1-0" / "1：0" 等
    if pd.isna(s):
        return np.nan, np.nan
    s = str(s).strip()
    if s in ("", "nan", "None"):
        return np.nan, np.nan
    for sep in [":", "：", "-", "–", "—"]:
        if sep in s:
            parts = s.split(sep)
            if len(parts) >= 2:
                try:
                    hg = float(str(parts[0]).strip())
                    ag = float(str(parts[1]).strip())
                    return hg, ag
                except Exception:
                    return np.nan, np.nan
    return np.nan, np.nan


def ensure_goals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "home_goals" not in df.columns or "away_goals" not in df.columns:
        # 尝试从 full_score_raw 解析
        if "full_score_raw" in df.columns:
            hg, ag = zip(*df["full_score_raw"].map(_parse_score_str))
            df["home_goals"] = pd.to_numeric(pd.Series(hg), errors="coerce")
            df["away_goals"] = pd.to_numeric(pd.Series(ag), errors="coerce")
        else:
            df["home_goals"] = np.nan
            df["away_goals"] = np.nan
    else:
        df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
        df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")
    return df


def parse_date(df: pd.DataFrame, date_col: str = "jc_date") -> pd.DataFrame:
    df = df.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df


def pick_team_key(row, home=True):
    # 优先 fbref，否则中文名
    if home:
        a = row.get("home_team_fbref")
        b = row.get("home_team_cn")
    else:
        a = row.get("away_team_fbref")
        b = row.get("away_team_cn")
    if pd.notna(a) and str(a).strip() != "":
        return str(a).strip()
    if pd.notna(b) and str(b).strip() != "":
        return str(b).strip()
    return "UNKNOWN"


def add_team_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["home_team_key"] = df.apply(lambda r: pick_team_key(r, True), axis=1)
    df["away_team_key"] = df.apply(lambda r: pick_team_key(r, False), axis=1)
    return df


def filter_train_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 剔除取消/无效
    if "bonus_isCancel" in df.columns:
        df = df[df["bonus_isCancel"].fillna(0).astype(int) != 1]
    # 必须有比分
    df = df[pd.notna(df["home_goals"]) & pd.notna(df["away_goals"])]
    return df


def parse_handicap(x):
    """
    支持：
    - 数值: -0.5, 1, 0
    - 字符串: "0/0.5", "-0.5/1", "0.25"
    规则：'a/b' 取平均 (a+b)/2
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return np.nan
    s = s.replace(" ", "")
    if "/" in s:
        parts = s.split("/")
        try:
            a = float(parts[0])
            b = float(parts[1])
            return (a + b) / 2.0
        except Exception:
            return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def add_hcp_line_numeric(df: pd.DataFrame, col_candidates: list[str]) -> pd.DataFrame:
    df = df.copy()
    line = None
    for c in col_candidates:
        if c in df.columns:
            line = df[c]
            break
    if line is None:
        df["hcp_line_used"] = np.nan
    else:
        df["hcp_line_used"] = line.map(parse_handicap)
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    y_1x2: 0主胜 1平 2客胜
    y_hhad: 0让球主胜 1让球平 2让球客胜 (使用 hcp_line_used)
    """
    df = df.copy()
    hg = df["home_goals"].astype(float)
    ag = df["away_goals"].astype(float)

    y1 = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))
    df["y_1x2"] = y1

    hcp = df.get("hcp_line_used", np.nan)
    adj = hg + pd.to_numeric(hcp, errors="coerce")
    y2 = np.where(adj > ag, 0, np.where(adj == ag, 1, 2))
    df["y_hhad"] = y2
    return df
```

---

## 6) src/fp/features.py

```python
from __future__ import annotations

from dataclasses import dataclass
import json
import numpy as np
import pandas as pd


def odds_to_implied_probs(odds_h, odds_d, odds_a):
    """
    输入赔率（SP），输出去水后的隐含概率。
    任一赔率缺失/<=1 返回 NaN。
    """
    oh, od, oa = odds_h, odds_d, odds_a
    if any(pd.isna([oh, od, oa])):
        return np.nan, np.nan, np.nan
    if (oh <= 1) or (od <= 1) or (oa <= 1):
        return np.nan, np.nan, np.nan
    inv = np.array([1.0/oh, 1.0/od, 1.0/oa], dtype=float)
    s = inv.sum()
    if s <= 0:
        return np.nan, np.nan, np.nan
    p = inv / s
    return float(p[0]), float(p[1]), float(p[2])


def pick_odds_triplet(df: pd.DataFrame, candidates: list[tuple[str, str, str]]):
    for a, b, c in candidates:
        if a in df.columns and b in df.columns and c in df.columns:
            # 不是全空就用
            if df[[a, b, c]].notna().any().any():
                return a, b, c
    return None


@dataclass
class PreprocessArtifacts:
    feature_cols: list[str]
    schema: dict


class FeatureBuilder:
    """
    训练时：
      - build_features(df, fit=True) 会确定 feature_cols 并保存 schema
    预测时：
      - build_features(df, fit=False) 会对齐 feature_cols（缺的补 NaN）
    """

    def __init__(self, max_goals: int = 7):
        self.max_goals = max_goals
        self.artifacts: PreprocessArtifacts | None = None

    def _basic_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        # fbref 差分（若列不存在自动 NaN）
        pairs = [
            ("home_xG", "away_xG", "xG_diff"),
            ("home_xGA", "away_xGA", "xGA_diff"),
            ("home_GF", "away_GF", "GF_diff"),
            ("home_GA", "away_GA", "GA_diff"),
            ("home_Pts", "away_Pts", "Pts_diff"),
            ("home_def_strength", "away_def_strength", "def_strength_diff"),
            ("home_off_buildup", "away_off_buildup", "off_buildup_diff"),
            ("home_tempo", "away_tempo", "tempo_diff_model"),
        ]
        for a, b, name in pairs:
            aa = pd.to_numeric(df[a], errors="coerce") if a in df.columns else np.nan
            bb = pd.to_numeric(df[b], errors="coerce") if b in df.columns else np.nan
            out[name] = aa - bb

        # 原始 tempo_diff 若存在则优先
        if "tempo_diff" in df.columns:
            out["tempo_diff_raw"] = pd.to_numeric(df["tempo_diff"], errors="coerce")

        # 联赛、比赛状态可做简单编码（ML 里用 one-hot 不如交给 LGBM 类别特征；这里先数值化为缺失）
        # 注意：若要加入类别特征，建议 CatBoost；这里保持纯数值可跑通

        return out

    def _market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        from fp.config import ODDS_SPF_CANDIDATES, ODDS_HCP_CANDIDATES

        spf_cols = pick_odds_triplet(df, ODDS_SPF_CANDIDATES)
        hcp_cols = pick_odds_triplet(df, ODDS_HCP_CANDIDATES)

        # SPF odds + implied probs
        if spf_cols is not None:
            h, d, a = spf_cols
            out["spf_odds_h"] = pd.to_numeric(df[h], errors="coerce")
            out["spf_odds_d"] = pd.to_numeric(df[d], errors="coerce")
            out["spf_odds_a"] = pd.to_numeric(df[a], errors="coerce")
            probs = df[[h, d, a]].apply(
                lambda r: odds_to_implied_probs(pd.to_numeric(r[h], errors="coerce"),
                                               pd.to_numeric(r[d], errors="coerce"),
                                               pd.to_numeric(r[a], errors="coerce")),
                axis=1, result_type="expand"
            )
            probs.columns = ["mkt_p_h", "mkt_p_d", "mkt_p_a"]
            out = pd.concat([out, probs], axis=1)
        else:
            out["spf_odds_h"] = np.nan
            out["spf_odds_d"] = np.nan
            out["spf_odds_a"] = np.nan
            out["mkt_p_h"] = np.nan
            out["mkt_p_d"] = np.nan
            out["mkt_p_a"] = np.nan

        # HHAD odds（不一定用于 mkt_p，主要做特征）
        if hcp_cols is not None:
            h, d, a = hcp_cols
            out["hcp_odds_h"] = pd.to_numeric(df[h], errors="coerce")
            out["hcp_odds_d"] = pd.to_numeric(df[d], errors="coerce")
            out["hcp_odds_a"] = pd.to_numeric(df[a], errors="coerce")
        else:
            out["hcp_odds_h"] = np.nan
            out["hcp_odds_d"] = np.nan
            out["hcp_odds_a"] = np.nan

        # 缺失指示器：赔率缺失/FBRef缺失
        out["is_missing_mkt"] = out[["mkt_p_h", "mkt_p_d", "mkt_p_a"]].isna().any(axis=1).astype(int)

        fbref_cols = [c for c in df.columns if c.startswith("home_") or c.startswith("away_")]
        if len(fbref_cols) > 0:
            # 粗略：关键列缺失就认为 fbref 缺
            key_cols = [c for c in ["home_xG", "away_xG", "home_xGA", "away_xGA", "home_tempo", "away_tempo"] if c in df.columns]
            if key_cols:
                out["is_missing_fbref"] = df[key_cols].isna().any(axis=1).astype(int)
            else:
                out["is_missing_fbref"] = 1
        else:
            out["is_missing_fbref"] = 1

        return out

    def build_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        base = self._basic_numeric(df)
        mkt = self._market_features(df)
        X = pd.concat([base, mkt], axis=1)

        # 可选：加入“市场热度/心理误差”字段（若未来有、历史也有才会被保留到 feature_cols）
        optional = []
        for c in [
            "had_support_h", "had_support_d", "had_support_a",
            "had_prob_h", "had_prob_d", "had_prob_a",
            "had_psy_err",
            "hhad_support_h", "hhad_support_d", "hhad_support_a",
            "hhad_prob_h", "hhad_prob_d", "hhad_prob_a",
            "hhad_psy_err",
            "bonus_score"
        ]:
            if c in df.columns:
                optional.append(pd.to_numeric(df[c], errors="coerce").rename(c))
        if optional:
            X = pd.concat([X] + optional, axis=1)

        if fit:
            feature_cols = list(X.columns)
            schema = {
                "feature_cols": feature_cols,
            }
            self.artifacts = PreprocessArtifacts(feature_cols=feature_cols, schema=schema)
            return X
        else:
            assert self.artifacts is not None, "FeatureBuilder not fitted. Load preprocess artifacts first."
            for c in self.artifacts.feature_cols:
                if c not in X.columns:
                    X[c] = np.nan
            X = X[self.artifacts.feature_cols]
            return X

    def save_schema(self, path: str) -> None:
        assert self.artifacts is not None
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.artifacts.schema, f, ensure_ascii=False, indent=2)

    def load_schema(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        self.artifacts = PreprocessArtifacts(feature_cols=schema["feature_cols"], schema=schema)
```

---

## 7) src/fp/calibration.py

```python
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV


class ProbCalibrator:
    """
    对 multiclass 分类器做概率校准（isotonic 或 sigmoid）。
    """
    def __init__(self, method: str = "isotonic"):
        self.method = method
        self._cal: CalibratedClassifierCV | None = None

    def fit(self, clf, X_val, y_val):
        cal = CalibratedClassifierCV(clf, method=self.method, cv="prefit")
        cal.fit(X_val, y_val)
        self._cal = cal
        return self

    def predict_proba(self, X) -> np.ndarray:
        assert self._cal is not None
        return self._cal.predict_proba(X)
```

---

## 8) src/fp/models/__init__.py

```python
# models package
```

---

## 9) src/fp/models/market.py

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from fp.features import odds_to_implied_probs
from fp.config import ODDS_SPF_CANDIDATES


def market_probs_spf(df: pd.DataFrame) -> pd.DataFrame:
    """
    从 SPF 赔率计算去水隐含概率（若缺失则 NaN）
    """
    cols = None
    for h, d, a in ODDS_SPF_CANDIDATES:
        if h in df.columns and d in df.columns and a in df.columns and df[[h, d, a]].notna().any().any():
            cols = (h, d, a)
            break
    out = pd.DataFrame(index=df.index, columns=["mkt_p_h", "mkt_p_d", "mkt_p_a"], dtype=float)
    if cols is None:
        out[:] = np.nan
        return out
    h, d, a = cols
    def f(r):
        oh = pd.to_numeric(r[h], errors="coerce")
        od = pd.to_numeric(r[d], errors="coerce")
        oa = pd.to_numeric(r[a], errors="coerce")
        return odds_to_implied_probs(oh, od, oa)
    tmp = df[[h, d, a]].apply(f, axis=1, result_type="expand")
    tmp.columns = out.columns
    return tmp
```

---

## 10) src/fp/models/elo.py

```python
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class EloConfig:
    k: float = 20.0
    home_adv: float = 60.0  # 主场加分
    init_rating: float = 1500.0


class EloModel:
    """
    简化足球 Elo（支持平局）：
    - 先计算 home win 的 logistic 概率
    - 平局概率用联赛平均 draw_rate + 随强弱差衰减
    """
    def __init__(self, cfg: EloConfig | None = None):
        self.cfg = cfg or EloConfig()
        self.ratings: dict[tuple[str, str], float] = {}  # (league, team_key) -> rating
        self.draw_rate_by_league: dict[str, float] = {}
        self.fitted = False

    def _get(self, league: str, team: str) -> float:
        return self.ratings.get((league, team), self.cfg.init_rating)

    def _set(self, league: str, team: str, r: float) -> None:
        self.ratings[(league, team)] = float(r)

    @staticmethod
    def _logistic(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    def fit(self, df: pd.DataFrame,
            league_col="league_name_jc",
            home_team_col="home_team_key",
            away_team_col="away_team_key",
            y_col="y_1x2"):
        # 估 draw_rate
        if league_col in df.columns:
            for lg, g in df.groupby(league_col):
                self.draw_rate_by_league[str(lg)] = float((g[y_col] == 1).mean())
        else:
            self.draw_rate_by_league["__GLOBAL__"] = float((df[y_col] == 1).mean())

        # 按时间顺序更新
        df2 = df.sort_values("jc_date")
        for _, r in df2.iterrows():
            lg = str(r.get(league_col, "__GLOBAL__"))
            ht = str(r[home_team_col])
            at = str(r[away_team_col])
            y = int(r[y_col])

            Rh = self._get(lg, ht)
            Ra = self._get(lg, at)

            # 期望主胜（不含平）
            diff = (Rh + self.cfg.home_adv - Ra) / 400.0
            p_win = self._logistic(np.log(10) * diff)  # 类似 Elo

            # 平局率：draw_rate * (1 - alpha*abs(p-0.5))^2
            d0 = self.draw_rate_by_league.get(lg, self.draw_rate_by_league.get("__GLOBAL__", 0.25))
            alpha = 1.5
            p_draw = float(np.clip(d0 * (1.0 - alpha * abs(p_win - 0.5))**2, 0.05, 0.45))
            p_home = (1 - p_draw) * p_win
            p_away = (1 - p_draw) * (1 - p_win)

            # 实际得分（胜=1 平=0.5 负=0）
            Sh = 1.0 if y == 0 else (0.5 if y == 1 else 0.0)
            Eh = p_home + 0.5 * p_draw

            delta = self.cfg.k * (Sh - Eh)
            self._set(lg, ht, Rh + delta)
            self._set(lg, at, Ra - delta)

        self.fitted = True
        return self

    def predict_proba(self, df: pd.DataFrame,
                      league_col="league_name_jc",
                      home_team_col="home_team_key",
                      away_team_col="away_team_key") -> np.ndarray:
        assert self.fitted
        probs = []
        for _, r in df.iterrows():
            lg = str(r.get(league_col, "__GLOBAL__"))
            ht = str(r[home_team_col])
            at = str(r[away_team_col])

            Rh = self._get(lg, ht)
            Ra = self._get(lg, at)
            diff = (Rh + self.cfg.home_adv - Ra) / 400.0
            p_win = self._logistic(np.log(10) * diff)

            d0 = self.draw_rate_by_league.get(lg, self.draw_rate_by_league.get("__GLOBAL__", 0.25))
            alpha = 1.5
            p_draw = float(np.clip(d0 * (1.0 - alpha * abs(p_win - 0.5))**2, 0.05, 0.45))

            p_home = (1 - p_draw) * p_win
            p_away = (1 - p_draw) * (1 - p_win)
            probs.append([p_home, p_draw, p_away])
        return np.array(probs, dtype=float)

    def features(self, df: pd.DataFrame,
                 league_col="league_name_jc",
                 home_team_col="home_team_key",
                 away_team_col="away_team_key") -> pd.DataFrame:
        assert self.fitted
        out = pd.DataFrame(index=df.index)
        Rh = df.apply(lambda r: self._get(str(r.get(league_col, "__GLOBAL__")), str(r[home_team_col])), axis=1)
        Ra = df.apply(lambda r: self._get(str(r.get(league_col, "__GLOBAL__")), str(r[away_team_col])), axis=1)
        out["elo_home"] = Rh.astype(float)
        out["elo_away"] = Ra.astype(float)
        out["elo_diff"] = out["elo_home"] - out["elo_away"]
        return out
```

---

## 11) src/fp/models/dc_poisson.py

```python
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln


@dataclass
class PoissonConfig:
    ridge: float = 1e-3
    max_iter: int = 400
    min_matches_per_league: int = 300
    # 时间衰减：weight = exp(-decay_per_day * days_ago)
    decay_per_day: float = 0.002


def _poisson_logpmf(k, lam):
    # log(P(K=k)) = k log lam - lam - log(k!)
    return k * np.log(lam + 1e-12) - lam - gammaln(k + 1)


def dc_tau(x, y, lam_h, lam_a, rho):
    """
    Dixon-Coles 低比分相关修正 tau(x,y)
    只对 (0,0)(0,1)(1,0)(1,1) 修正
    """
    if x == 0 and y == 0:
        return 1.0 - (lam_h * lam_a * rho)
    if x == 0 and y == 1:
        return 1.0 + (lam_h * rho)
    if x == 1 and y == 0:
        return 1.0 + (lam_a * rho)
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


class DCPoissonModel:
    """
    攻防泊松 + DC 低比分修正 rho
    log lam_home = mu + home_adv + att_home - def_away
    log lam_away = mu + att_away - def_home

    约束：sum(att)=0, sum(def)=0 通过“去均值”实现。
    """
    def __init__(self, cfg: PoissonConfig | None = None):
        self.cfg = cfg or PoissonConfig()
        self.params_by_league: dict[str, dict] = {}
        self.global_params: dict | None = None
        self.fitted = False

    def _fit_one(self, df: pd.DataFrame, league: str) -> dict:
        teams = sorted(set(df["home_team_key"]).union(set(df["away_team_key"])))
        tid = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        # 数据
        hg = df["home_goals"].astype(int).to_numpy()
        ag = df["away_goals"].astype(int).to_numpy()
        ht = df["home_team_key"].map(tid).to_numpy()
        at = df["away_team_key"].map(tid).to_numpy()

        # 时间衰减
        max_date = df["jc_date"].max()
        days_ago = (max_date - df["jc_date"]).dt.days.clip(lower=0).to_numpy()
        w = np.exp(-self.cfg.decay_per_day * days_ago)

        # 参数向量：mu, home_adv, att[n], def[n], rho
        # 通过去均值约束：在计算时 att/def 去均值
        x0 = np.zeros(2 + 2*n + 1, dtype=float)
        x0[0] = np.log(max(df[["home_goals", "away_goals"]].stack().mean(), 1e-3))
        x0[1] = 0.1  # home_adv

        def unpack(x):
            mu = x[0]
            ha = x[1]
            att = x[2:2+n]
            dff = x[2+n:2+2*n]
            rho = np.tanh(x[-1]) * 0.2  # 限制 rho 范围 ~[-0.2,0.2]
            # 约束：去均值
            att = att - att.mean()
            dff = dff - dff.mean()
            return mu, ha, att, dff, rho

        ridge = self.cfg.ridge

        def nll(x):
            mu, ha, att, dff, rho = unpack(x)
            lam_h = np.exp(mu + ha + att[ht] - dff[at])
            lam_a = np.exp(mu + att[at] - dff[ht])

            # base loglik
            ll = _poisson_logpmf(hg, lam_h) + _poisson_logpmf(ag, lam_a)

            # DC tau 修正（仅低比分）
            tau = np.ones_like(lam_h)
            mask = (hg <= 1) & (ag <= 1)
            if mask.any():
                for i in np.where(mask)[0]:
                    tau[i] = dc_tau(int(hg[i]), int(ag[i]), lam_h[i], lam_a[i], rho)
            ll = ll + np.log(np.clip(tau, 1e-9, None))

            # 权重
            llw = (w * ll).sum()

            # ridge 正则（不惩罚 mu）
            pen = ridge * (x[1:] @ x[1:])
            return -llw + pen

        res = minimize(nll, x0, method="L-BFGS-B", options={"maxiter": self.cfg.max_iter})
        x = res.x
        mu, ha, att, dff, rho = unpack(x)

        return {
            "league": league,
            "teams": teams,
            "tid": tid,
            "mu": float(mu),
            "home_adv": float(ha),
            "att": att.astype(float),
            "def": dff.astype(float),
            "rho": float(rho),
            "success": bool(res.success),
            "fun": float(res.fun),
        }

    def fit(self, df: pd.DataFrame, league_col="league_name_jc"):
        df = df.copy()
        if league_col not in df.columns:
            df[league_col] = "__GLOBAL__"

        # 全局参数（兜底）
        self.global_params = self._fit_one(df, "__GLOBAL__")

        # 联赛参数（样本足够就拟合）
        self.params_by_league = {}
        for lg, g in df.groupby(league_col):
            lg = str(lg)
            if len(g) >= self.cfg.min_matches_per_league:
                self.params_by_league[lg] = self._fit_one(g, lg)

        self.fitted = True
        return self

    def _get_params(self, league: str) -> dict:
        if league in self.params_by_league:
            return self.params_by_league[league]
        assert self.global_params is not None
        return self.global_params

    def predict_lambda(self, df: pd.DataFrame, league_col="league_name_jc") -> pd.DataFrame:
        assert self.fitted
        out = pd.DataFrame(index=df.index, columns=["lambda_home", "lambda_away", "rho_used"], dtype=float)

        for idx, r in df.iterrows():
            lg = str(r.get(league_col, "__GLOBAL__"))
            p = self._get_params(lg)
            tid = p["tid"]
            mu = p["mu"]
            ha = p["home_adv"]
            att = p["att"]
            dff = p["def"]
            rho = p["rho"]

            ht = str(r["home_team_key"])
            at = str(r["away_team_key"])
            # 新队伍未见过：当作均值(0)
            i = tid.get(ht, None)
            j = tid.get(at, None)
            att_h = att[i] if i is not None else 0.0
            def_h = dff[i] if i is not None else 0.0
            att_a = att[j] if j is not None else 0.0
            def_a = dff[j] if j is not None else 0.0

            lam_h = float(np.exp(mu + ha + att_h - def_a))
            lam_a = float(np.exp(mu + att_a - def_h))

            out.loc[idx, "lambda_home"] = lam_h
            out.loc[idx, "lambda_away"] = lam_a
            out.loc[idx, "rho_used"] = rho

        return out
```

---

## 12) src/fp/models/ml.py

```python
from __future__ import annotations

import numpy as np
import lightgbm as lgb


def train_lgbm_multiclass(X_train, y_train, X_val, y_val, num_class=3, seed=42):
    params = {
        "objective": "multiclass",
        "num_class": num_class,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_data_in_leaf": 40,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 2.0,
        "metric": "multi_logloss",
        "seed": seed,
        "verbose": -1,
    }
    dtr = lgb.Dataset(X_train, label=y_train)
    dva = lgb.Dataset(X_val, label=y_val, reference=dtr)
    clf = lgb.train(
        params,
        dtr,
        num_boost_round=5000,
        valid_sets=[dva],
        callbacks=[
            lgb.early_stopping(stopping_rounds=150, verbose=False),
        ],
    )
    return clf


def predict_proba_lgbm(clf, X) -> np.ndarray:
    p = clf.predict(X)
    p = np.asarray(p, dtype=float)
    # shape: (n,3)
    return p
```

---

## 13) src/fp/models/ensemble.py

```python
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _logloss(y_true, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    n = len(y_true)
    return -np.log(p[np.arange(n), y_true]).mean()


def fit_weights(y_true, probs_dict: dict[str, np.ndarray]) -> dict:
    """
    probs_dict: name -> (n,3) array
    学习非负权重和为1，使 logloss 最小
    """
    names = list(probs_dict.keys())
    P = np.stack([probs_dict[k] for k in names], axis=0)  # (m,n,3)
    m = P.shape[0]

    x0 = np.ones(m) / m

    def objective(w):
        w = np.clip(w, 0, 1)
        s = w.sum()
        if s <= 0:
            w = np.ones_like(w) / len(w)
        else:
            w = w / s
        p = np.tensordot(w, P, axes=(0, 0))  # (n,3)
        return _logloss(y_true, p)

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * m
    res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 200})
    w = np.clip(res.x, 0, 1)
    w = w / max(w.sum(), 1e-12)
    return {"names": names, "weights": w.tolist(), "success": bool(res.success), "fun": float(res.fun)}


def combine_dynamic(probs: dict[str, np.ndarray], weights: dict[str, float], availability: dict[str, np.ndarray]) -> np.ndarray:
    """
    availability[name] = (n,) 1可用 0不可用
    对每行比赛：将不可用模型权重置0，再归一化加权。
    """
    names = list(weights.keys())
    n = next(iter(probs.values())).shape[0]
    out = np.zeros((n, 3), dtype=float)

    W = np.array([weights[k] for k in names], dtype=float)  # (m,)
    P = np.stack([probs[k] for k in names], axis=0)         # (m,n,3)
    A = np.stack([availability[k] for k in names], axis=0)  # (m,n)

    for i in range(n):
        w = W * A[:, i]
        s = w.sum()
        if s <= 0:
            w = np.ones_like(w) / len(w)
        else:
            w = w / s
        out[i] = np.tensordot(w, P[:, i, :], axes=(0, 0))
    return out


def save_ensemble_cfg(cfg: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def load_ensemble_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
```

---

## 14) src/fp/simulate.py

```python
from __future__ import annotations

import numpy as np
import pandas as pd
from math import exp
from fp.models.dc_poisson import dc_tau


def poisson_pmf(k: int, lam: float) -> float:
    # 简单 pmf（用于构建矩阵）
    from math import factorial
    return (lam ** k) * exp(-lam) / factorial(k)


def score_matrix(lam_h: float, lam_a: float, rho: float = 0.0, max_goals: int = 7) -> np.ndarray:
    """
    返回 (max_goals+1, max_goals+1) 的比分概率矩阵（含 DC tau 修正，未含尾部截断校正会再归一化）
    """
    M = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
    for i in range(max_goals + 1):
        pi = poisson_pmf(i, lam_h)
        for j in range(max_goals + 1):
            pj = poisson_pmf(j, lam_a)
            p = pi * pj
            if i <= 1 and j <= 1:
                p *= max(dc_tau(i, j, lam_h, lam_a, rho), 1e-9)
            M[i, j] = p
    s = M.sum()
    if s > 0:
        M /= s
    return M


def probs_from_matrix(M: np.ndarray) -> dict:
    max_goals = M.shape[0] - 1
    p_home = float(np.triu(M, 1).sum())  # i>j
    p_draw = float(np.trace(M))
    p_away = float(np.tril(M, -1).sum())

    # over/under 2.5
    total = np.fromfunction(lambda i, j: i + j, M.shape, dtype=int)
    p_over25 = float(M[total >= 3].sum())
    p_under25 = 1.0 - p_over25

    # BTTS
    i = np.arange(max_goals + 1)
    j = np.arange(max_goals + 1)
    I, J = np.meshgrid(i, j, indexing="ij")
    p_btts = float(M[(I >= 1) & (J >= 1)].sum())

    return {
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "p_over25": p_over25,
        "p_under25": p_under25,
        "p_btts_yes": p_btts,
    }


def top_scores(M: np.ndarray, topn: int = 3):
    flat = []
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            flat.append((i, j, float(M[i, j])))
    flat.sort(key=lambda x: x[2], reverse=True)
    return flat[:topn]


def hhad_probs_from_matrix(M: np.ndarray, hcp_line: float) -> tuple[float, float, float]:
    """
    让球胜平负概率（以“主队 + hcp_line”对比客队为准）
    """
    if hcp_line is None or (isinstance(hcp_line, float) and np.isnan(hcp_line)):
        return np.nan, np.nan, np.nan
    max_goals = M.shape[0] - 1
    pH = pD = pA = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            adj = i + hcp_line
            if adj > j:
                pH += M[i, j]
            elif adj == j:
                pD += M[i, j]
            else:
                pA += M[i, j]
    s = pH + pD + pA
    if s > 0:
        return float(pH/s), float(pD/s), float(pA/s)
    return np.nan, np.nan, np.nan
```

---

## 15) src/fp/betting.py

```python
from __future__ import annotations

import numpy as np
import pandas as pd


def ev_from_odds_prob(odds: float, prob: float) -> float:
    if pd.isna(odds) or pd.isna(prob) or odds <= 1 or prob <= 0:
        return np.nan
    return float(odds * prob - 1.0)


def kelly_fraction(odds: float, prob: float, cap: float = 0.05, kelly_scale: float = 0.5) -> float:
    """
    Kelly: f* = (b p - q)/b, b=odds-1
    返回缩放后的下注比例，并做上限 cap
    """
    if pd.isna(odds) or pd.isna(prob) or odds <= 1 or prob <= 0 or prob >= 1:
        return 0.0
    b = odds - 1.0
    q = 1.0 - prob
    f = (b * prob - q) / b
    f = max(0.0, float(f))
    f = f * kelly_scale
    return float(min(f, cap))


def pick_best_bet(row: pd.Series,
                  min_ev: float = 0.02,
                  min_prob: float = 0.05):
    """
    在 SPF 三项中找 best EV（也可以扩展到 HHAD/大小球）
    """
    cand = []
    for key, odds_col, p_col in [
        ("SPF_HOME", "spf_odds_h", "pred_prob_home"),
        ("SPF_DRAW", "spf_odds_d", "pred_prob_draw"),
        ("SPF_AWAY", "spf_odds_a", "pred_prob_away"),
    ]:
        odds = row.get(odds_col, np.nan)
        prob = row.get(p_col, np.nan)
        if pd.notna(prob) and prob >= min_prob and pd.notna(odds) and odds > 1:
            ev = ev_from_odds_prob(float(odds), float(prob))
            kf = kelly_fraction(float(odds), float(prob))
            cand.append((key, ev, kf))
    if not cand:
        return "NO_BET", np.nan, 0.0
    cand.sort(key=lambda x: (x[1] if pd.notna(x[1]) else -999), reverse=True)
    best = cand[0]
    if pd.isna(best[1]) or best[1] < min_ev:
        return "NO_BET", float(best[1]) if pd.notna(best[1]) else np.nan, float(best[2])
    return best[0], float(best[1]), float(best[2])
```

---

## 16) src/fp/train.py  （训练：生成 artifacts）

```python
from __future__ import annotations

import argparse
import json
import os
import joblib
import numpy as np
import pandas as pd

from fp.io import read_csv, Paths
from fp.config import ARTIFACTS, HCP_LINE_CANDIDATES
from fp.clean import (
    parse_date, ensure_goals, add_team_keys, filter_train_rows,
    add_hcp_line_numeric, add_targets
)
from fp.features import FeatureBuilder
from fp.models.elo import EloModel, EloConfig
from fp.models.dc_poisson import DCPoissonModel, PoissonConfig
from fp.models.ml import train_lgbm_multiclass, predict_proba_lgbm
from fp.calibration import ProbCalibrator
from fp.models.ensemble import fit_weights, save_ensemble_cfg


def time_split(df: pd.DataFrame, date_col="jc_date", val_ratio=0.2):
    df = df.sort_values(date_col)
    n = len(df)
    cut = int(n * (1 - val_ratio))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--artifacts_dir", required=True)
    args = ap.parse_args()

    paths = Paths(artifacts_dir=args.artifacts_dir)

    df = read_csv(args.train_csv)
    df = parse_date(df, "jc_date")
    df = ensure_goals(df)
    df = add_team_keys(df)
    df = add_hcp_line_numeric(df, HCP_LINE_CANDIDATES)
    df = filter_train_rows(df)
    df = add_targets(df)

    # 分割
    tr, va = time_split(df, "jc_date", val_ratio=0.2)

    # 1) Elo
    elo = EloModel(EloConfig(k=18.0, home_adv=60.0))
    elo.fit(tr)

    tr_elo_feat = elo.features(tr)
    va_elo_feat = elo.features(va)

    # 2) DC Poisson
    pm = DCPoissonModel(PoissonConfig(min_matches_per_league=300))
    pm.fit(tr)

    tr_lam = pm.predict_lambda(tr)
    va_lam = pm.predict_lambda(va)

    # 3) FeatureBuilder（基础特征 + 市场特征）
    fb = FeatureBuilder()
    X_tr = fb.build_features(tr, fit=True)
    X_va = fb.build_features(va, fit=False)

    # 拼入二级特征（Elo + lam）
    X_tr = pd.concat([X_tr, tr_elo_feat, tr_lam[["lambda_home", "lambda_away"]]], axis=1)
    X_va = pd.concat([X_va, va_elo_feat, va_lam[["lambda_home", "lambda_away"]]], axis=1)

    # 保存 schema（加入二级特征后，更新 feature_cols）
    fb.artifacts.feature_cols = list(X_tr.columns)
    fb.save_schema(paths.a(ARTIFACTS["feature_schema"]))

    # 4) ML 训练：1x2
    y_tr_1x2 = tr["y_1x2"].astype(int).to_numpy()
    y_va_1x2 = va["y_1x2"].astype(int).to_numpy()
    clf1 = train_lgbm_multiclass(X_tr, y_tr_1x2, X_va, y_va_1x2, num_class=3, seed=42)

    # 5) ML 训练：hhad
    y_tr_hhad = tr["y_hhad"].astype(int).to_numpy()
    y_va_hhad = va["y_hhad"].astype(int).to_numpy()
    clf2 = train_lgbm_multiclass(X_tr, y_tr_hhad, X_va, y_va_hhad, num_class=3, seed=43)

    # 6) 校准
    cal1 = ProbCalibrator(method="isotonic").fit(clf1, X_va, y_va_1x2)
    cal2 = ProbCalibrator(method="isotonic").fit(clf2, X_va, y_va_hhad)

    # 7) 融合权重学习（用验证集）
    # 7.1 ML 概率
    p_ml_1x2 = cal1.predict_proba(X_va)

    # 7.2 Poisson 从 lambda 构建 1x2（简单用解析矩阵）
    from fp.simulate import score_matrix, probs_from_matrix
    p_ps = []
    for lh, la, rho in zip(va_lam["lambda_home"], va_lam["lambda_away"], va_lam["rho_used"]):
        M = score_matrix(float(lh), float(la), float(rho), max_goals=7)
        d = probs_from_matrix(M)
        p_ps.append([d["p_home"], d["p_draw"], d["p_away"]])
    p_ps = np.array(p_ps, dtype=float)

    # 7.3 Elo 概率
    p_elo = elo.predict_proba(va)

    # 7.4 市场概率（用 feature 里的 mkt_p）
    mkt = X_va[["mkt_p_h", "mkt_p_d", "mkt_p_a"]].to_numpy(dtype=float)

    # 缺失处理：市场缺就先用均匀（权重学习时避免 NaN）
    mkt2 = mkt.copy()
    nanrow = np.isnan(mkt2).any(axis=1)
    mkt2[nanrow] = np.array([1/3, 1/3, 1/3])

    probs_dict = {
        "ml": p_ml_1x2,
        "poisson": p_ps,
        "elo": p_elo,
        "market": mkt2,
    }
    wcfg = fit_weights(y_va_1x2, probs_dict)

    # 经验性动态规则：如果 market 缺失，则该场 market 权重=0；fbref 缺失则 ml 权重打折
    ensemble_cfg = {
        "base_weights": {k: float(w) for k, w in zip(wcfg["names"], wcfg["weights"])},
        "rules": {
            "if_missing_market": {"market": 0.0},
            "if_missing_fbref": {"ml_multiplier": 0.6},
        },
        "fit_info": wcfg,
    }
    save_ensemble_cfg(ensemble_cfg, paths.a(ARTIFACTS["ensemble_cfg"]))

    # 8) 保存 artifacts
    joblib.dump(fb, paths.a(ARTIFACTS["preprocess"]))
    joblib.dump(clf1, paths.a(ARTIFACTS["model_ml_1x2"]))
    joblib.dump(clf2, paths.a(ARTIFACTS["model_ml_hhad"]))
    joblib.dump(cal1, paths.a(ARTIFACTS["calib_1x2"]))
    joblib.dump(cal2, paths.a(ARTIFACTS["calib_hhad"]))
    joblib.dump(elo, paths.a(ARTIFACTS["model_elo"]))
    joblib.dump(pm, paths.a(ARTIFACTS["model_poisson"]))

    meta = {
        "train_csv": os.path.abspath(args.train_csv),
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "date_min": str(df["jc_date"].min()),
        "date_max": str(df["jc_date"].max()),
    }
    with open(paths.a(ARTIFACTS["train_meta"]), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[OK] Training finished.")
    print("Artifacts saved to:", os.path.abspath(args.artifacts_dir))
    print("Ensemble base weights:", ensemble_cfg["base_weights"])


if __name__ == "__main__":
    main()
```

---

## 17) src/fp/predict.py  （预测：加载 artifacts，输出到 datasets/）

```python
from __future__ import annotations

import argparse
import os
import joblib
import numpy as np
import pandas as pd

from fp.io import read_csv, write_csv, glob_paths, Paths
from fp.config import ARTIFACTS, HCP_LINE_CANDIDATES, MAX_GOALS
from fp.clean import parse_date, ensure_goals, add_team_keys, add_hcp_line_numeric
from fp.models.ensemble import load_ensemble_cfg, combine_dynamic
from fp.simulate import score_matrix, probs_from_matrix, top_scores, hhad_probs_from_matrix
from fp.betting import ev_from_odds_prob, kelly_fraction, pick_best_bet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--future_glob", required=True)
    ap.add_argument("--artifacts_dir", required=True)
    ap.add_argument("--out_dir", required=True)  # 按你的要求可直接 datasets
    args = ap.parse_args()

    paths = Paths(artifacts_dir=args.artifacts_dir, out_dir=args.out_dir)

    # load artifacts
    fb = joblib.load(paths.a(ARTIFACTS["preprocess"]))
    clf1 = joblib.load(paths.a(ARTIFACTS["model_ml_1x2"]))
    clf2 = joblib.load(paths.a(ARTIFACTS["model_ml_hhad"]))
    cal1 = joblib.load(paths.a(ARTIFACTS["calib_1x2"]))
    cal2 = joblib.load(paths.a(ARTIFACTS["calib_hhad"]))
    elo = joblib.load(paths.a(ARTIFACTS["model_elo"]))
    pm = joblib.load(paths.a(ARTIFACTS["model_poisson"]))
    ens = load_ensemble_cfg(paths.a(ARTIFACTS["ensemble_cfg"]))

    base_weights = ens["base_weights"]
    rules = ens.get("rules", {})

    files = glob_paths(args.future_glob)
    if not files:
        raise FileNotFoundError(f"No files matched: {args.future_glob}")

    for fp in files:
        df = read_csv(fp)
        df = parse_date(df, "jc_date")
        df = ensure_goals(df)  # 未来可能没比分，保留 NaN 不影响
        df = add_team_keys(df)
        df = add_hcp_line_numeric(df, HCP_LINE_CANDIDATES)

        # feature
        X = fb.build_features(df, fit=False)

        # add elo + poisson lambdas
        elo_feat = elo.features(df)
        lam = pm.predict_lambda(df)
        X2 = pd.concat([X, elo_feat, lam[["lambda_home", "lambda_away"]]], axis=1)

        # predict probs
        p_ml_1x2 = cal1.predict_proba(X2)
        p_ml_hhad = cal2.predict_proba(X2)

        # poisson-derived 1x2 + 派生
        p_ps_1x2 = []
        p_over25 = []
        p_btts = []
        p_hhad_from_poisson = []
        mode_score = []
        top3 = []
        for lh, la, rho, hcp_line in zip(
            lam["lambda_home"].astype(float),
            lam["lambda_away"].astype(float),
            lam["rho_used"].astype(float),
            df["hcp_line_used"].astype(float),
        ):
            M = score_matrix(float(lh), float(la), float(rho), max_goals=MAX_GOALS)
            d = probs_from_matrix(M)
            p_ps_1x2.append([d["p_home"], d["p_draw"], d["p_away"]])
            p_over25.append(d["p_over25"])
            p_btts.append(d["p_btts_yes"])

            hh = hhad_probs_from_matrix(M, float(hcp_line) if pd.notna(hcp_line) else np.nan)
            p_hhad_from_poisson.append(list(hh))

            ts = top_scores(M, topn=3)
            mode_score.append(f"{ts[0][0]}-{ts[0][1]}")
            top3.append("|".join([f"{a}-{b}:{p:.3f}" for a, b, p in ts]))

        p_ps_1x2 = np.array(p_ps_1x2, dtype=float)
        p_hhad_from_poisson = np.array(p_hhad_from_poisson, dtype=float)

        # elo probs
        p_elo = elo.predict_proba(df)

        # market probs（在 X 已经有 mkt_p）
        p_mkt = X[["mkt_p_h", "mkt_p_d", "mkt_p_a"]].to_numpy(dtype=float)

        # availability flags
        n = len(df)
        avail = {
            "ml": np.ones(n, dtype=float),
            "poisson": np.ones(n, dtype=float),
            "elo": np.ones(n, dtype=float),
            "market": (~np.isnan(p_mkt).any(axis=1)).astype(float),
        }
        # 如果 fbref 缺失，ml 权重乘折扣
        is_missing_fbref = X.get("is_missing_fbref", pd.Series([1]*n, index=df.index)).to_numpy()
        ml_mult = rules.get("if_missing_fbref", {}).get("ml_multiplier", 1.0)
        avail["ml"] = np.where(is_missing_fbref == 1, ml_mult, 1.0)

        probs_pool = {
            "ml": p_ml_1x2,
            "poisson": p_ps_1x2,
            "elo": p_elo,
            "market": np.where(np.isnan(p_mkt), np.array([1/3, 1/3, 1/3]), p_mkt),
        }
        p_ens_1x2 = combine_dynamic(probs_pool, base_weights, avail)

        # 让球胜平负：这里给两套
        # (1) ML hhad
        # (2) poisson 让球（已算）
        # 简单融合：各 0.5（可后续也做权重学习）
        p_ens_hhad = 0.5 * p_ml_hhad + 0.5 * np.where(np.isnan(p_hhad_from_poisson), p_ml_hhad, p_hhad_from_poisson)

        # 组装输出
        out = df.copy()
        out["pred_prob_home"] = p_ens_1x2[:, 0]
        out["pred_prob_draw"] = p_ens_1x2[:, 1]
        out["pred_prob_away"] = p_ens_1x2[:, 2]

        out["pred_lambda_home"] = lam["lambda_home"].astype(float)
        out["pred_lambda_away"] = lam["lambda_away"].astype(float)
        out["pred_prob_over25"] = np.array(p_over25, dtype=float)
        out["pred_prob_under25"] = 1.0 - out["pred_prob_over25"]
        out["pred_prob_btts_yes"] = np.array(p_btts, dtype=float)

        out["pred_score_mode"] = mode_score
        out["pred_score_top3"] = top3

        # 市场概率（可能为空）
        out["mkt_prob_home"] = p_mkt[:, 0]
        out["mkt_prob_draw"] = p_mkt[:, 1]
        out["mkt_prob_away"] = p_mkt[:, 2]

        out["edge_home"] = out["pred_prob_home"] - out["mkt_prob_home"]
        out["edge_draw"] = out["pred_prob_draw"] - out["mkt_prob_draw"]
        out["edge_away"] = out["pred_prob_away"] - out["mkt_prob_away"]

        # EV/Kelly 基于 SPF odds（X 里有 spf_odds_*）
        out["spf_odds_h"] = X.get("spf_odds_h", np.nan)
        out["spf_odds_d"] = X.get("spf_odds_d", np.nan)
        out["spf_odds_a"] = X.get("spf_odds_a", np.nan)

        out["ev_home"] = out.apply(lambda r: ev_from_odds_prob(r["spf_odds_h"], r["pred_prob_home"]), axis=1)
        out["ev_draw"] = out.apply(lambda r: ev_from_odds_prob(r["spf_odds_d"], r["pred_prob_draw"]), axis=1)
        out["ev_away"] = out.apply(lambda r: ev_from_odds_prob(r["spf_odds_a"], r["pred_prob_away"]), axis=1)

        out["kelly_home"] = out.apply(lambda r: kelly_fraction(r["spf_odds_h"], r["pred_prob_home"]), axis=1)
        out["kelly_draw"] = out.apply(lambda r: kelly_fraction(r["spf_odds_d"], r["pred_prob_draw"]), axis=1)
        out["kelly_away"] = out.apply(lambda r: kelly_fraction(r["spf_odds_a"], r["pred_prob_away"]), axis=1)

        # 推荐投注（SPF）
        best = out.apply(pick_best_bet, axis=1, result_type="expand")
        best.columns = ["best_bet_market", "best_bet_ev", "best_bet_kelly_stake"]
        out = pd.concat([out, best], axis=1)

        # 让球胜平负输出
        out["pred_hhad_prob_home"] = p_ens_hhad[:, 0]
        out["pred_hhad_prob_draw"] = p_ens_hhad[:, 1]
        out["pred_hhad_prob_away"] = p_ens_hhad[:, 2]

        # 输出文件名
        base = os.path.basename(fp).replace(".csv", "")
        out_path = paths.o(f"{base}_pred.csv")
        write_csv(out, out_path)

        print("[OK] Predicted:", os.path.basename(fp), "->", out_path)


if __name__ == "__main__":
    main()
```

---

## 18) README.md（使用说明，建议直接放根目录）

```md
# Football Predictor (JC + FBRef Hybrid)

## 1. 功能
- 训练一次保存 artifacts（可复用）
- 对 jc_fbref_future_*.csv 生成预测与投注建议
- 输出：胜平负/让球胜平负概率、lambda、Top比分、大小球、EV/Kelly、推荐投注项

## 2. 数据
- 训练集：datasets/jc_fbref_bonus_support_2025-01-01_to_2026-01-17.csv
- 预测集：datasets/jc_fbref_future_*.csv
- 输出：datasets/jc_fbref_future_*_pred.csv

## 3. 安装
```bash
pip install -r requirements.txt
```

## 4. 训练（只需一次）
```bash
python -m fp.train \
  --train_csv datasets/jc_fbref_bonus_support_2025-01-01_to_2026-01-17.csv \
  --artifacts_dir artifacts
```

## 5. 预测（复用 artifacts）
```bash
python -m fp.predict \
  --future_glob "datasets/jc_fbref_future_*.csv" \
  --artifacts_dir artifacts \
  --out_dir datasets
```

## 6. 输出字段（节选）
- pred_prob_home/draw/away：融合后的胜平负概率
- pred_lambda_home/away：泊松期望进球
- pred_score_mode / pred_score_top3：最可能比分与Top3
- pred_prob_over25 / pred_prob_btts_yes：大小球与BTTS
- mkt_prob_*：去水后的市场隐含概率（赔率缺失则为空）
- ev_* / kelly_*：EV 与凯利下注比例（默认半凯利、上限5%）
- best_bet_market：SPF 推荐投注（若无价值则 NO_BET）

## 7. 风险提示
仅供数据分析，不保证盈利。请做好资金管理。
```

---

# 你下一步要我补齐/增强的点（建议你确认）
你现在这套脚本能跑通并产出预测，但为了更贴合你真实数据，我建议你确认两件事，我再给你“增强版”补丁：

1) **hcp_line 的符号定义**：你数据里 `-1` 是“主让1球”还是“主受1球”？  
   - 当前实现按“主队 + hcp_line 与客队比较”来判定，让 `-1` 表示主队要减1（即主让1球）。如果你数据定义相反，需要统一修正。

2) 你希望投注建议覆盖哪些玩法：  
   - 目前示例只对 **SPF** 做 EV/Kelly 推荐；我可以扩展到 **HHAD/大小球(2.5)/比分Top3组合**，并输出更完整的“价值洼地清单”。

你回复这两点（尤其是让球符号），我就把对应脚本做成“更贴近你数据口径”的最终版。

`-1` 表示**主让1球**的话，我之前脚本里让球判断采用的是：

- 让球后主队进球 = `home_goals + hcp_line`
- 与 `away_goals` 比较

因此 `hcp_line = -1` 时就是 `home_goals - 1`，**口径正确，无需改符号**。

你要“投注建议覆盖胜平负 + 让球胜平负”，下面给出需要替换/新增的**具体脚本内容**（只改 `betting.py` 和 `predict.py` 即可），实现：
- SPF + HHAD 都计算 EV / Kelly
- 在两种玩法里挑选“最优一项”作为 `best_bet_market`

---

## 1) 替换：`src/fp/betting.py`

```python
from __future__ import annotations

import numpy as np
import pandas as pd


def ev_from_odds_prob(odds: float, prob: float) -> float:
    """
    EV = odds * prob - 1
    """
    if pd.isna(odds) or pd.isna(prob) or odds <= 1 or prob <= 0:
        return np.nan
    return float(odds * prob - 1.0)


def kelly_fraction(odds: float, prob: float, cap: float = 0.05, kelly_scale: float = 0.5) -> float:
    """
    Kelly: f* = (b p - q)/b, b=odds-1
    返回缩放后的下注比例，并做上限 cap
    默认：半凯利 + 单项最大5%资金
    """
    if pd.isna(odds) or pd.isna(prob) or odds <= 1 or prob <= 0 or prob >= 1:
        return 0.0
    b = odds - 1.0
    q = 1.0 - prob
    f = (b * prob - q) / b
    f = max(0.0, float(f))
    f *= kelly_scale
    return float(min(f, cap))


def pick_best_bet_spf_hhad(
    row: pd.Series,
    min_ev: float = 0.02,
    min_prob: float = 0.05,
):
    """
    在 SPF + HHAD 共6个选项里挑 best EV，若 best EV < min_ev 则 NO_BET
    输出: (best_bet_market, best_bet_ev, best_bet_kelly_stake)
    """
    candidates = []

    # SPF
    spf_items = [
        ("SPF_HOME", "spf_odds_h", "pred_prob_home"),
        ("SPF_DRAW", "spf_odds_d", "pred_prob_draw"),
        ("SPF_AWAY", "spf_odds_a", "pred_prob_away"),
    ]
    for key, odds_col, p_col in spf_items:
        odds = row.get(odds_col, np.nan)
        prob = row.get(p_col, np.nan)
        if pd.notna(prob) and prob >= min_prob and pd.notna(odds) and odds > 1:
            ev = ev_from_odds_prob(float(odds), float(prob))
            kf = kelly_fraction(float(odds), float(prob))
            candidates.append((key, ev, kf))

    # HHAD（让球胜平负）
    hhad_items = [
        ("HHAD_HOME", "hhad_odds_h", "pred_hhad_prob_home"),
        ("HHAD_DRAW", "hhad_odds_d", "pred_hhad_prob_draw"),
        ("HHAD_AWAY", "hhad_odds_a", "pred_hhad_prob_away"),
    ]
    for key, odds_col, p_col in hhad_items:
        odds = row.get(odds_col, np.nan)
        prob = row.get(p_col, np.nan)
        if pd.notna(prob) and prob >= min_prob and pd.notna(odds) and odds > 1:
            ev = ev_from_odds_prob(float(odds), float(prob))
            kf = kelly_fraction(float(odds), float(prob))
            candidates.append((key, ev, kf))

    if not candidates:
        return "NO_BET", np.nan, 0.0

    candidates.sort(key=lambda x: (x[1] if pd.notna(x[1]) else -999), reverse=True)
    best_key, best_ev, best_kf = candidates[0]

    if pd.isna(best_ev) or best_ev < min_ev:
        return "NO_BET", float(best_ev) if pd.notna(best_ev) else np.nan, float(best_kf)
    return best_key, float(best_ev), float(best_kf)
```

---

## 2) 修改：`src/fp/predict.py`

把原来的 `pick_best_bet` 替换为 `pick_best_bet_spf_hhad`，并补上 HHAD 的 odds/EV/Kelly 输出列。

下面给出“关键替换块”，你按位置替换即可（整文件其它部分不变）。

### 2.1 修改 import（文件顶部）
把：
```python
from fp.betting import ev_from_odds_prob, kelly_fraction, pick_best_bet
```
改为：
```python
from fp.betting import ev_from_odds_prob, kelly_fraction, pick_best_bet_spf_hhad
```

### 2.2 在输出 odds 时加入 HHAD odds
在你现有代码这段之后：

```python
out["spf_odds_h"] = X.get("spf_odds_h", np.nan)
out["spf_odds_d"] = X.get("spf_odds_d", np.nan)
out["spf_odds_a"] = X.get("spf_odds_a", np.nan)
```

追加：

```python
# HHAD odds（来自 FeatureBuilder 里的 hcp_odds_*）
out["hhad_odds_h"] = X.get("hcp_odds_h", np.nan)
out["hhad_odds_d"] = X.get("hcp_odds_d", np.nan)
out["hhad_odds_a"] = X.get("hcp_odds_a", np.nan)
```

### 2.3 先输出让球概率，再计算 HHAD 的 EV/Kelly
确保你已经有：

```python
out["pred_hhad_prob_home"] = p_ens_hhad[:, 0]
out["pred_hhad_prob_draw"] = p_ens_hhad[:, 1]
out["pred_hhad_prob_away"] = p_ens_hhad[:, 2]
```

在其后追加：

```python
# HHAD EV/Kelly
out["hhad_ev_home"] = out.apply(lambda r: ev_from_odds_prob(r["hhad_odds_h"], r["pred_hhad_prob_home"]), axis=1)
out["hhad_ev_draw"] = out.apply(lambda r: ev_from_odds_prob(r["hhad_odds_d"], r["pred_hhad_prob_draw"]), axis=1)
out["hhad_ev_away"] = out.apply(lambda r: ev_from_odds_prob(r["hhad_odds_a"], r["pred_hhad_prob_away"]), axis=1)

out["hhad_kelly_home"] = out.apply(lambda r: kelly_fraction(r["hhad_odds_h"], r["pred_hhad_prob_home"]), axis=1)
out["hhad_kelly_draw"] = out.apply(lambda r: kelly_fraction(r["hhad_odds_d"], r["pred_hhad_prob_draw"]), axis=1)
out["hhad_kelly_away"] = out.apply(lambda r: kelly_fraction(r["hhad_odds_a"], r["pred_hhad_prob_away"]), axis=1)
```

### 2.4 用 SPF+HHAD 一起选 best bet
把原来的：

```python
best = out.apply(pick_best_bet, axis=1, result_type="expand")
best.columns = ["best_bet_market", "best_bet_ev", "best_bet_kelly_stake"]
out = pd.concat([out, best], axis=1)
```

改为：

```python
best = out.apply(pick_best_bet_spf_hhad, axis=1, result_type="expand")
best.columns = ["best_bet_market", "best_bet_ev", "best_bet_kelly_stake"]
out = pd.concat([out, best], axis=1)
```

---

## 3) 结果解释（你会在输出里看到什么）
- `best_bet_market` 可能是：
  - `SPF_HOME / SPF_DRAW / SPF_AWAY`
  - `HHAD_HOME / HHAD_DRAW / HHAD_AWAY`
  - `NO_BET`
- `best_bet_ev`：该推荐项的期望收益（>0 才有价值，默认阈值 0.02）
- `best_bet_kelly_stake`：建议下注资金占比（半凯利+5%封顶）

---

如果你希望“优先推荐让球或胜平负”（例如同等 EV 时优先 SPF），我也可以把 tie-break 规则加进去。你只要告诉我偏好顺序即可。