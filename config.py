import os
from pathlib import Path

# 基础路径配置
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "datasets"
PREDICT_DATA_DIR = ROOT_DIR / "predict_data"
OUTPUT_DIR = ROOT_DIR / "outputs"

# 子目录配置
MODEL_DIR = OUTPUT_DIR / "models"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
BACKTEST_DIR = OUTPUT_DIR / "backtest"
BETTING_DIR = OUTPUT_DIR / "betting_recommendations"

# 创建目录
for dir_path in [MODEL_DIR, PREDICTION_DIR, BACKTEST_DIR, BETTING_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 数据文件配置
HISTORY_DATA_FILE = DATA_DIR / "jc_fbref_bonus_support_feature_2025-01-01_to_2026-01-21.csv"
PREDICT_DATA_PATTERN = PREDICT_DATA_DIR / "jc_today_unplayed_for_predict_*.csv"

# 模型配置
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
MODEL_SAVE_FORMAT = "pkl"  # 模型保存格式
TARGETS = {
    "1x2": "result_1x2",  # 胜平负
    "hcp_1x2": "hcp_result",  # 让球胜平负（后续生成）
    "over_under": "ou25_result",  # 大小球2.5
    "score": ["home_goals", "away_goals"]  # 比分
}

# 投注配置
KELLEY_MIN_PROB = 0.05  # 凯利公式最小概率阈值
EV_MIN_VALUE = 0.01     # 期望价值最小阈值
COMMISSION_RATE = 0.02  # 投注佣金率