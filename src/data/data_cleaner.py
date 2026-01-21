import pandas as pd
import numpy as np
from loguru import logger

def clean_history_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗历史数据：
    1. 处理空值（区分竞彩未开放和fbref数据缺失）
    2. 格式转换（日期、数值型字段）
    3. 生成目标变量（让球胜平负、比分差等）
    4. 过滤无效数据
    """
    df_clean = df.copy()
    
    # 1. 日期格式转换
    df_clean["jc_date"] = pd.to_datetime(df_clean["jc_date"], errors="coerce")
    
    # 2. 数值型字段转换
    numeric_cols = [
        "home_GF", "home_GA", "home_xG", "home_xGA", "home_Pts",
        "away_GF", "away_GA", "away_xG", "away_xGA", "away_Pts",
        "hcp_line_now", "hcp_line_open", "hcp_sp_home_now", "hcp_sp_away_now",
        "spf_sp_home_close", "spf_sp_draw_close", "spf_sp_away_close",
        "home_goals", "away_goals", "goal_diff", "total_goals"
    ]
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
    
    # 3. 处理空值策略
    # - 竞彩数据空值（未开放投注）：标记并过滤
    # 注意：poolStatus可能为空，所以只检查必要的赔率列
    betting_cols = ["hcp_sp_home_now", "spf_sp_home_close"]
    df_clean["is_betting_available"] = df_clean[betting_cols].notnull().all(axis=1)
    
    # - fbref数据空值：填充为联赛均值（保留缺失标记）
    fbref_cols = ["home_xG", "away_xG", "home_def_strength", "away_def_strength"]
    df_clean["is_fbref_available"] = df_clean[fbref_cols].notnull().all(axis=1)
    
    # 按联赛填充fbref缺失值
    for col in fbref_cols:
        if col in df_clean.columns:
            league_means = df_clean.groupby("league_name_jc")[col].transform("mean")
            df_clean[col] = df_clean[col].fillna(league_means)
    
    # 4. 生成让球胜平负目标变量
    if all(col in df_clean.columns for col in ["home_goals", "away_goals", "hcp_line_close"]):
        # 让球线通常是主队让球（如0.5表示主队让0.5球）
        df_clean["hcp_goal_diff"] = df_clean["home_goals"] - df_clean["away_goals"] - df_clean["hcp_line_close"]
        df_clean["hcp_result"] = np.where(
            df_clean["hcp_goal_diff"] > 0, "H",  # 让球胜
            np.where(df_clean["hcp_goal_diff"] == 0, "D", "A")  # 让球平/负
        )
    
    # 5. 过滤无效数据（无比赛结果、无投注数据）
    df_clean = df_clean[
        (df_clean["result_1x2"].notnull()) & 
        (df_clean["is_betting_available"]) &
        (df_clean["jc_date"].notnull())
    ].reset_index(drop=True)
    
    logger.info(f"数据清洗完成，清洗后数据量：{len(df_clean)} 行（原数据：{len(df)} 行）")
    return df_clean

def clean_predict_data(df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    """清洗待预测数据（对齐历史数据格式）"""
    df_clean = df.copy()
    
    # 1. 日期格式转换
    df_clean["jc_date"] = pd.to_datetime(df_clean["jc_date"], errors="coerce")
    
    # 2. 数值型字段转换
    numeric_cols = [
        "home_GF", "home_GA", "home_xG", "home_xGA", "home_Pts",
        "away_GF", "away_GA", "away_xG", "away_xGA", "away_Pts",
        "hcp_line_now", "hcp_line_open", "hcp_sp_home_now", "hcp_sp_away_now",
        "spf_sp_home_close", "spf_sp_draw_close", "spf_sp_away_close"
    ]
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
    
    # 3. 处理空值策略
    # - 竞彩数据空值（未开放投注）：标记并过滤
    # 注意：poolStatus可能为空，所以只检查必要的赔率列
    betting_cols = ["hcp_sp_home_now", "spf_sp_home_close"]
    df_clean["is_betting_available"] = df_clean[betting_cols].notnull().all(axis=1)
    
    # - fbref数据空值：填充为联赛均值（保留缺失标记）
    fbref_cols = ["home_xG", "away_xG", "home_def_strength", "away_def_strength"]
    df_clean["is_fbref_available"] = df_clean[fbref_cols].notnull().all(axis=1)
    
    # 按联赛填充fbref缺失值
    for col in fbref_cols:
        if col in df_clean.columns:
            league_means = df_clean.groupby("league_name_jc")[col].transform("mean")
            df_clean[col] = df_clean[col].fillna(league_means)
    
    # 4. 过滤无效数据（无投注数据）
    df_clean = df_clean[
        (df_clean["is_betting_available"]) &
        (df_clean["jc_date"].notnull())
    ].reset_index(drop=True)
    
    # 5. 对齐字段（确保和历史数据字段一致）
    common_cols = list(set(history_df.columns) & set(df_clean.columns))
    df_clean = df_clean[common_cols]
    
    logger.info(f"待预测数据清洗完成，数据量：{len(df_clean)} 行")
    return df_clean
