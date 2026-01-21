import pandas as pd
import numpy as np
from loguru import logger

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    构建预测特征：
    1. 基础统计特征（场均进球、胜率、攻防效率）
    2. 市场赔率特征（赔率转换概率、赔率差、凯利指数）
    3. fbref进阶特征（xG差值、攻防强度比）
    4. 时间/联赛特征（主场优势、联赛强度）
    """
    df_features = df.copy()
    
    # 1. 基础统计特征
    # 主队vs客队进球差
    df_features["home_away_goal_diff"] = df_features["home_goal_avg_cnt"] - df_features["away_goal_avg_cnt"]
    # 主队胜率 - 客队胜率
    df_features["win_rate_diff"] = df_features["home_win_rate"] - df_features["away_win_rate"]
    # 总进球预期
    df_features["expected_total_goals"] = df_features["home_GF"] + df_features["away_GF"]
    
    # 2. 市场赔率特征（转换为概率，去除佣金）
    def odds_to_prob(odds: pd.Series, commission: float = 0.02) -> pd.Series:
        """赔率转概率（处理佣金）"""
        prob = 1 / odds
        prob_sum = prob.sum()
        return prob / (prob_sum * (1 - commission))
    
    # 胜平负概率
    if all(col in df_features.columns for col in ["spf_sp_home_close", "spf_sp_draw_close", "spf_sp_away_close"]):
        df_features["prob_home"] = odds_to_prob(df_features["spf_sp_home_close"])
        df_features["prob_draw"] = odds_to_prob(df_features["spf_sp_draw_close"])
        df_features["prob_away"] = odds_to_prob(df_features["spf_sp_away_close"])
    
    # 3. fbref进阶特征
    if all(col in df_features.columns for col in ["home_xG", "away_xG"]):
        # xG差值
        df_features["xg_diff"] = df_features["home_xG"] - df_features["away_xG"]
        # 攻防效率比
        df_features["home_attack_defense_ratio"] = df_features["home_off_buildup"] / df_features["home_def_strength"]
        df_features["away_attack_defense_ratio"] = df_features["away_off_buildup"] / df_features["away_def_strength"]
        # 节奏差值
        df_features["tempo_diff_normalized"] = df_features["tempo_diff"] / df_features["tempo_diff"].abs().max()
    
    # 4. 分类特征编码（联赛、球队）
    # 联赛编码
    df_features["league_encoded"] = df_features["league_name_jc"].astype("category").cat.codes
    # 主场标记（固定为1，客队为0）
    df_features["is_home"] = 1
    
    # 5. 过滤无效特征
    # 移除高基数分类特征（如match_id）和冗余特征
    drop_cols = ["match_id", "match_num", "home_team_cn", "away_team_cn", "full_score_raw", "half_score_raw"]
    df_features = df_features.drop(columns=[col for col in drop_cols if col in df_features.columns])
    
    # 填充剩余空值
    df_features = df_features.fillna(df_features.median(numeric_only=True))
    
    logger.info(f"特征工程完成，生成特征数：{len(df_features.columns)}")
    return df_features

def split_features_target(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """分离特征和目标变量"""
    # 过滤非数值型特征
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    # 移除目标变量和冗余列
    features = [col for col in numeric_features if col not in [target_col, "home_goals", "away_goals", "goal_diff", "total_goals"]]
    
    X = df[features]
    y = df[target_col]
    
    logger.info(f"特征矩阵形状：{X.shape}，目标变量形状：{y.shape}")
    return X, y