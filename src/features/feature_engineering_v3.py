import pandas as pd
import numpy as np
from loguru import logger

def create_features_v3(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    构建预测特征（精确版本）：
    只移除真正的结果相关数据，保留所有赛前可用的特征
    
    Parameters:
    -----------
    df : pd.DataFrame
        原始数据
    is_training : bool
        是否为训练数据（训练数据包含目标变量，预测数据不包含）
    
    Returns:
    --------
    pd.DataFrame
        特征数据
    """
    df_features = df.copy()
    
    # 1. 基础球队历史统计特征（赛前可用）
    # 这些是基于球队历史表现的数据，不是本场比赛的结果
    if all(col in df_features.columns for col in ["home_goal_avg_cnt", "away_goal_avg_cnt"]):
        df_features["home_away_goal_avg_diff"] = df_features["home_goal_avg_cnt"] - df_features["away_goal_avg_cnt"]
    
    if all(col in df_features.columns for col in ["home_win_rate", "away_win_rate"]):
        df_features["win_rate_diff"] = df_features["home_win_rate"] - df_features["away_win_rate"]
    
    # 2. 市场赔率特征（赛前可用）
    def odds_to_prob(odds: pd.Series, commission: float = 0.02) -> pd.Series:
        """赔率转概率（处理佣金）"""
        prob = 1 / odds
        prob_sum = prob.sum()
        return prob / (prob_sum * (1 - commission))
    
    # 胜平负赔率概率
    if all(col in df_features.columns for col in ["spf_sp_home_close", "spf_sp_draw_close", "spf_sp_away_close"]):
        df_features["prob_home"] = odds_to_prob(df_features["spf_sp_home_close"])
        df_features["prob_draw"] = odds_to_prob(df_features["spf_sp_draw_close"])
        df_features["prob_away"] = odds_to_prob(df_features["spf_sp_away_close"])
    
    # 让球赔率概率
    if all(col in df_features.columns for col in ["hcp_sp_home_close", "hcp_sp_draw_close", "hcp_sp_away_close"]):
        df_features["hcp_prob_home"] = odds_to_prob(df_features["hcp_sp_home_close"])
        df_features["hcp_prob_draw"] = odds_to_prob(df_features["hcp_sp_draw_close"])
        df_features["hcp_prob_away"] = odds_to_prob(df_features["hcp_sp_away_close"])
    
    # 3. fbref进阶特征（赛前可用）
    # 这些是基于球队赛季表现的数据，不是本场比赛的结果
    if all(col in df_features.columns for col in ["home_xG", "away_xG"]):
        df_features["xg_diff"] = df_features["home_xG"] - df_features["away_xG"]
    
    if all(col in df_features.columns for col in ["home_def_strength", "away_def_strength"]):
        df_features["def_strength_diff"] = df_features["home_def_strength"] - df_features["away_def_strength"]
    
    if all(col in df_features.columns for col in ["home_off_buildup", "away_off_buildup"]):
        df_features["off_buildup_diff"] = df_features["home_off_buildup"] - df_features["away_off_buildup"]
    
    # 4. 分类特征编码（赛前可用）
    # 联赛编码
    if "league_name_jc" in df_features.columns:
        df_features["league_encoded"] = df_features["league_name_jc"].astype("category").cat.codes
    
    # 5. 时间特征（赛前可用）
    if "jc_date" in df_features.columns:
        df_features["jc_date"] = pd.to_datetime(df_features["jc_date"], errors="coerce")
        df_features["month"] = df_features["jc_date"].dt.month
        df_features["day_of_week"] = df_features["jc_date"].dt.dayofweek
        df_features["is_weekend"] = df_features["day_of_week"].isin([5, 6]).astype(int)
    
    # 6. 精确移除真正的结果相关列（数据泄露）
    # 根据分析，只有这些列在预测数据中为空，但在历史数据中不为空
    true_result_cols = [
        # 比分相关（预测数据为空）
        "full_score_raw", "half_score_raw",
        # 投注状态相关（预测数据为空）
        "bettingSingle", "poolStatus", "bonus_score",
        # 比赛结果（在预测数据中不存在）
        "result_1x2", "goal_diff", "total_goals", "ou25_result",
        "home_goals", "away_goals",
        # 让球结果相关（在预测数据中不存在）
        "hcp_goal_diff", "hcp_result",
    ]
    
    # 如果是训练模式，保留目标变量result_1x2
    if is_training:
        # 从移除列表中排除result_1x2
        if "result_1x2" in true_result_cols:
            true_result_cols.remove("result_1x2")
    
    # 只移除存在的列
    cols_to_drop = [col for col in true_result_cols if col in df_features.columns]
    if cols_to_drop:
        df_features = df_features.drop(columns=cols_to_drop)
        logger.info(f"移除了 {len(cols_to_drop)} 个真正的结果相关特征")
        logger.debug(f"移除的列: {cols_to_drop}")
    
    # 7. 移除高基数分类特征和ID列（但不是所有特征）
    drop_cols = [
        "match_id", "match_num", "home_team_cn", "away_team_cn",
        "home_team_fbref", "away_team_fbref", "jc_date",
        "business_date", "match_time", "league_name_jc",
        "home_team_abb", "away_team_abb", "home_team_id", "away_team_id",
        "league_abb_name", "league_id", "match_num_str",
    ]
    
    # 只移除存在的列
    cols_to_drop = [col for col in drop_cols if col in df_features.columns]
    if cols_to_drop:
        df_features = df_features.drop(columns=cols_to_drop)
    
    # 8. 填充剩余空值
    df_features = df_features.fillna(df_features.median(numeric_only=True))
    
    logger.info(f"特征工程完成，生成特征数：{len(df_features.columns)}")
    return df_features

def split_features_target_v3(df: pd.DataFrame, target_col: str = "result_1x2") -> tuple[pd.DataFrame, pd.Series]:
    """
    分离特征和目标变量（精确版本）
    
    注意：这个函数假设df已经通过create_features_v3处理过，
    并且不包含任何结果相关的特征（除了目标变量）
    """
    # 首先确保目标变量存在
    if target_col not in df.columns:
        raise ValueError(f"目标变量 '{target_col}' 不在数据中")
    
    # 获取所有数值型特征
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 从特征中移除目标变量
    features = [col for col in numeric_features if col != target_col]
    
    X = df[features]
    y = df[target_col]
    
    logger.info(f"特征矩阵形状：{X.shape}，目标变量形状：{y.shape}")
    return X, y
