import pandas as pd
import glob
from pathlib import Path
from loguru import logger
from config import HISTORY_DATA_FILE, PREDICT_DATA_PATTERN

def load_history_data() -> pd.DataFrame:
    """加载历史数据并做基础检查"""
    try:
        df = pd.read_csv(HISTORY_DATA_FILE, encoding="utf-8")
        logger.info(f"加载历史数据成功，数据量：{len(df)} 行，字段数：{len(df.columns)}")
        
        # 基础数据检查
        logger.info(f"数据时间范围：{df['jc_date'].min()} 至 {df['jc_date'].max()}")
        logger.info(f"联赛数量：{df['league_name_jc'].nunique()}")
        logger.info(f"空值统计：\n{df.isnull().sum()[df.isnull().sum() > 0]}")
        
        return df
    except Exception as e:
        logger.error(f"加载历史数据失败：{e}")
        raise

def load_predict_data() -> pd.DataFrame:
    """加载待预测数据（匹配通配符）"""
    try:
        predict_files = glob.glob(str(PREDICT_DATA_PATTERN))
        if not predict_files:
            logger.warning("未找到待预测数据文件")
            return pd.DataFrame()
        
        # 合并所有待预测文件
        df_list = []
        for file in predict_files:
            df = pd.read_csv(file, encoding="utf-8")
            df_list.append(df)
            logger.info(f"加载待预测文件 {Path(file).name}，数据量：{len(df)} 行")
        
        predict_df = pd.concat(df_list, ignore_index=True)
        logger.info(f"合并后待预测数据量：{len(predict_df)} 行")
        return predict_df
    except Exception as e:
        logger.error(f"加载待预测数据失败：{e}")
        raise