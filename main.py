from loguru import logger
from src.prediction.predictor import FootballPredictor
from src.backtest.backtest_analyzer import BacktestAnalyzer
from src.features.feature_engineering import split_features_target
from config import PREDICTION_DIR, BETTING_DIR

def main():
    """主运行函数"""
    # 初始化预测器
    predictor = FootballPredictor()
    
    # 步骤1：加载并预处理数据
    logger.info("=== 步骤1：数据加载与预处理 ===")
    predictor.load_and_preprocess_data()
    
    # 步骤2：训练所有模型
    logger.info("\n=== 步骤2：模型训练 ===")
    predictor.train_all_models()
    
    # 步骤3：预测未进行的比赛
    logger.info("\n=== 步骤3：比赛预测 ===")
    predict_results, value_bets = predictor.predict_unplayed_matches()
    
    # 增加容错：如果无预测数据，跳过后续保存逻辑
    if predict_results.empty:
        logger.warning("无预测结果，跳过预测结果保存")
    else:
        logger.info(f"预测结果已保存到：{PREDICTION_DIR / 'match_predictions.csv'}")
        if not value_bets.empty:
            logger.info(f"价值投注推荐已保存到：{BETTING_DIR / 'value_betting_recommendations.csv'}")
    
    # 步骤4：策略回测
    logger.info("\n=== 步骤4：策略回测 ===")
    backtest_analyzer = BacktestAnalyzer(initial_capital=10000.0)
    
    # 为历史数据添加投注分析
    history_df_with_features = predictor.features_df.copy()
    # 机器学习预测
    X_history, y_history = split_features_target(
        history_df_with_features, target_col="result_1x2"
    )
    _, ml_prob_history = predictor.ml_trainer.predict(X_history)
    
    # 构建历史数据的预测概率
    history_df_with_features["pred_home_prob"] = [p[0] for p in ml_prob_history]
    history_df_with_features["pred_draw_prob"] = [p[1] for p in ml_prob_history]
    history_df_with_features["pred_away_prob"] = [p[2] for p in ml_prob_history]
    
    # 投注分析
    pred_prob_dict = {
        "home": history_df_with_features["pred_home_prob"],
        "draw": history_df_with_features["pred_draw_prob"],
        "away": history_df_with_features["pred_away_prob"]
    }
    history_betting_df = predictor.betting_analyzer.analyze_value_bets(
        history_df_with_features, pred_prob_dict
    )
    
    # 回测价值投注策略
    backtest_analyzer.backtest_betting_strategy(
        history_betting_df, 
        bet_size=100.0, 
        strategy="value_bet"
    )
    
    logger.info("\n=== 项目运行完成 ===")
    logger.info(f"预测结果路径：predictions/match_predictions.csv（如有数据）")
    logger.info(f"价值投注推荐路径：betting_recommendations/value_betting_recommendations.csv（如有数据）")
    logger.info(f"回测结果路径：backtest/")

if __name__ == "__main__":
    main()
