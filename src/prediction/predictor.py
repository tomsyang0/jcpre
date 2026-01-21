import pandas as pd
from loguru import logger
from src.data.data_loader import load_history_data, load_predict_data
from src.data.data_cleaner import clean_history_data, clean_predict_data
from src.features.feature_engineering import create_features, split_features_target
from src.features.feature_engineering_v2 import create_features_v2, split_features_target_v2
from src.features.feature_engineering_v3 import create_features_v3, split_features_target_v3
from src.models.statistical_models import PoissonPredictor, EloPredictor, MonteCarloPredictor
from src.models.ml_models import MLModelTrainer
from src.models.betting_models import BettingAnalyzer
from config import PREDICTION_DIR, BETTING_DIR

class FootballPredictor:
    """足球比赛预测主类"""
    def __init__(self):
        # 初始化所有模型
        self.poisson_model = PoissonPredictor()
        self.elo_model = EloPredictor()
        self.monte_carlo_model = MonteCarloPredictor(n_simulations=10000)
        self.ml_trainer = MLModelTrainer(model_type="xgboost")
        self.betting_analyzer = BettingAnalyzer(commission_rate=0.02)
        
        # 数据存储
        self.history_df = None
        self.predict_df = None
        self.features_df = None
    
    def load_and_preprocess_data(self):
        """加载并预处理数据"""
        # 加载历史数据
        self.history_df = load_history_data()
        self.history_df = clean_history_data(self.history_df)
        
        # 加载待预测数据
        self.predict_df = load_predict_data()
        if not self.predict_df.empty:
            self.predict_df = clean_predict_data(self.predict_df, self.history_df)
        
        # 特征工程（使用精确版本）
        self.features_df = create_features_v3(self.history_df, is_training=True)
        if not self.predict_df.empty:
            self.predict_features_df = create_features_v3(self.predict_df, is_training=False)
    
    def train_all_models(self):
        """训练所有预测模型"""
        logger.info("开始训练统计模型...")
        # 训练统计模型
        self.poisson_model.fit(self.history_df)
        self.elo_model.fit(self.history_df)
        self.monte_carlo_model.fit(self.history_df)
        
        logger.info("开始训练机器学习模型...")
        # 训练机器学习模型（胜平负预测）- 使用精确版本
        X, y = split_features_target_v3(self.features_df, target_col="result_1x2")
        self.ml_trainer.fit(X, y)
        
        # 输出特征重要性
        feature_importance = self.ml_trainer.get_feature_importance(X.columns)
        feature_importance.to_csv(PREDICTION_DIR / "feature_importance.csv", index=False, encoding="utf-8")
        logger.info("特征重要性已保存")
    
    def predict_unplayed_matches(self):
        """预测未进行的比赛"""
        if self.predict_df.empty:
            logger.warning("无待预测数据")
            # 修复：返回空DataFrame而非None，避免解包错误
            return pd.DataFrame(), pd.DataFrame()
        
        # 1. 机器学习模型预测
        # 我们需要确保预测数据的特征与训练数据的特征完全一致
        # 首先获取训练时使用的特征
        X_train, _ = split_features_target_v3(self.features_df, target_col="result_1x2")
        train_features = X_train.columns.tolist()
        
        # 创建预测特征矩阵，只包含训练时使用的特征
        X_predict = pd.DataFrame()
        for feature in train_features:
            if feature in self.predict_features_df.columns:
                X_predict[feature] = self.predict_features_df[feature]
            else:
                # 如果特征在预测数据中不存在，填充为0
                X_predict[feature] = 0
                logger.warning(f"预测数据中缺少特征 '{feature}'，已填充为0")
        
        ml_predictions, ml_probabilities = self.ml_trainer.predict(X_predict)
        
        # 2. 统计模型预测
        poisson_probs = []
        elo_probs = []
        monte_carlo_probs = []
        
        for _, row in self.predict_df.iterrows():
            league = row["league_name_jc"]
            home_team = row["home_team_fbref"]
            away_team = row["away_team_fbref"]
            
            # 泊松模型预测
            _, poisson_prob = self.poisson_model.predict_score_prob(league, home_team, away_team)
            poisson_probs.append(poisson_prob)
            
            # 埃罗模型预测
            elo_prob = self.elo_model.predict(home_team, away_team)
            elo_probs.append(elo_prob)
            
            # 蒙特卡罗模拟
            mc_result = self.monte_carlo_model.simulate(league, home_team, away_team)
            monte_carlo_probs.append(mc_result["1x2_prob"])
        
        # 3. 融合预测概率（加权平均）
        fused_probs = []
        for i in range(len(ml_probabilities)):
            # 机器学习概率（权重0.5）
            ml_prob = {
                "home": ml_probabilities[i][0],
                "draw": ml_probabilities[i][1],
                "away": ml_probabilities[i][2]
            }
            
            # 统计模型平均概率（权重0.5）
            stat_prob = {
                "home": (poisson_probs[i]["home"] + elo_probs[i]["home"] + monte_carlo_probs[i]["home"]) / 3,
                "draw": (poisson_probs[i]["draw"] + elo_probs[i]["draw"] + monte_carlo_probs[i]["draw"]) / 3,
                "away": (poisson_probs[i]["away"] + elo_probs[i]["away"] + monte_carlo_probs[i]["away"]) / 3
            }
            
            # 融合概率
            fused_prob = {
                "home": 0.5 * ml_prob["home"] + 0.5 * stat_prob["home"],
                "draw": 0.5 * ml_prob["draw"] + 0.5 * stat_prob["draw"],
                "away": 0.5 * ml_prob["away"] + 0.5 * stat_prob["away"]
            }
            
            # 归一化
            total = fused_prob["home"] + fused_prob["draw"] + fused_prob["away"]
            fused_prob["home"] /= total
            fused_prob["draw"] /= total
            fused_prob["away"] /= total
            
            fused_probs.append(fused_prob)
        
        # 4. 投注价值分析
        predict_df_with_probs = self.predict_df.copy()
        predict_df_with_probs["ml_prediction"] = ml_predictions
        predict_df_with_probs["fused_home_prob"] = [p["home"] for p in fused_probs]
        predict_df_with_probs["fused_draw_prob"] = [p["draw"] for p in fused_probs]
        predict_df_with_probs["fused_away_prob"] = [p["away"] for p in fused_probs]
        
        # 价值投注分析
        pred_prob_dict = {
            "home": predict_df_with_probs["fused_home_prob"],
            "draw": predict_df_with_probs["fused_draw_prob"],
            "away": predict_df_with_probs["fused_away_prob"]
        }
        betting_df = self.betting_analyzer.analyze_value_bets(predict_df_with_probs, pred_prob_dict)
        
        # 保存预测结果
        betting_df.to_csv(PREDICTION_DIR / "match_predictions.csv", index=False, encoding="utf-8")
        
        # 筛选价值投注推荐
        value_bets_df = betting_df[betting_df["has_value_bet"]][[
            "jc_date", "league_name_jc", "home_team_cn", "away_team_cn",
            "fused_home_prob", "fused_draw_prob", "fused_away_prob",
            "spf_sp_home_close", "spf_sp_draw_close", "spf_sp_away_close",
            "ev_home", "ev_draw", "ev_away",
            "kelley_home", "kelley_draw", "kelley_away",
            "betting_recommendation"
        ]]
        value_bets_df.to_csv(BETTING_DIR / "value_betting_recommendations.csv", index=False, encoding="utf-8")
        
        logger.info(f"预测完成！")
        logger.info(f"共预测 {len(betting_df)} 场比赛")
        logger.info(f"价值投注推荐 {len(value_bets_df)} 场，已保存到 {BETTING_DIR}")
        
        return betting_df, value_bets_df
