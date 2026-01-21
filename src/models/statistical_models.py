import pandas as pd
import numpy as np
from scipy.stats import poisson, norm
from loguru import logger
import random

class PoissonPredictor:
    """泊松分布比分预测模型"""
    def __init__(self):
        self.home_attack_strength = {}
        self.home_defense_strength = {}
        self.away_attack_strength = {}
        self.away_defense_strength = {}
        self.league_avg_goals = {}
    
    def fit(self, df: pd.DataFrame):
        """训练泊松模型（按联赛计算攻防强度）"""
        for league in df["league_name_jc"].unique():
            league_df = df[df["league_name_jc"] == league]
            
            # 联赛平均进球数
            avg_home_goals = league_df["home_GF"].mean()
            avg_away_goals = league_df["away_GF"].mean()
            self.league_avg_goals[league] = (avg_home_goals, avg_away_goals)
            
            # 计算球队攻防强度
            for team in league_df["home_team_fbref"].unique():
                # 主队进攻强度 = 该队主场平均进球 / 联赛主场平均进球
                team_home_df = league_df[league_df["home_team_fbref"] == team]
                team_home_attack = team_home_df["home_GF"].mean() / avg_home_goals if avg_home_goals > 0 else 1.0
                
                # 主队防守强度 = 该队主场失球 / 联赛客场平均进球
                team_home_defense = team_home_df["away_GF"].mean() / avg_away_goals if avg_away_goals > 0 else 1.0
                
                self.home_attack_strength[(league, team)] = team_home_attack
                self.home_defense_strength[(league, team)] = team_home_defense
            
            for team in league_df["away_team_fbref"].unique():
                # 客队进攻强度 = 该队客场平均进球 / 联赛客场平均进球
                team_away_df = league_df[league_df["away_team_fbref"] == team]
                team_away_attack = team_away_df["away_GF"].mean() / avg_away_goals if avg_away_goals > 0 else 1.0
                
                # 客队防守强度 = 该队客场失球 / 联赛主场平均进球
                team_away_defense = team_away_df["home_GF"].mean() / avg_home_goals if avg_home_goals > 0 else 1.0
                
                self.away_attack_strength[(league, team)] = team_away_attack
                self.away_defense_strength[(league, team)] = team_away_defense
        
        logger.info("泊松模型训练完成")
    
    def predict_score_prob(self, league: str, home_team: str, away_team: str, max_goals: int = 5) -> pd.DataFrame:
        """预测比分概率矩阵"""
        # 获取联赛平均进球
        avg_home, avg_away = self.league_avg_goals.get(league, (1.5, 1.2))
        
        # 获取球队攻防强度（默认1.0）
        home_attack = self.home_attack_strength.get((league, home_team), 1.0)
        home_defense = self.home_defense_strength.get((league, home_team), 1.0)
        away_attack = self.away_attack_strength.get((league, away_team), 1.0)
        away_defense = self.away_defense_strength.get((league, away_team), 1.0)
        
        # 计算预期进球数
        lambda_home = avg_home * home_attack * away_defense
        lambda_away = avg_away * away_attack * home_defense
        
        # 生成比分概率矩阵
        score_probs = {}
        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                prob = poisson.pmf(home_goals, lambda_home) * poisson.pmf(away_goals, lambda_away)
                score_probs[(home_goals, away_goals)] = prob
        
        # 转换为DataFrame
        prob_df = pd.DataFrame([
            {"home_goals": h, "away_goals": a, "probability": p}
            for (h, a), p in score_probs.items()
        ])
        
        # 计算胜平负概率
        prob_home = prob_df[prob_df["home_goals"] > prob_df["away_goals"]]["probability"].sum()
        prob_draw = prob_df[prob_df["home_goals"] == prob_df["away_goals"]]["probability"].sum()
        prob_away = prob_df[prob_df["home_goals"] < prob_df["away_goals"]]["probability"].sum()
        
        return prob_df, {"home": prob_home, "draw": prob_draw, "away": prob_away}

class EloPredictor:
    """埃罗(Elo)预测模型"""
    def __init__(self, base_elo: int = 1500, k_factor: int = 32):
        self.base_elo = base_elo
        self.k_factor = k_factor
        self.team_elos = {}
    
    def fit(self, df: pd.DataFrame):
        """训练埃罗模型（按比赛更新Elo评分）"""
        # 初始化所有球队Elo
        all_teams = set(df["home_team_fbref"].unique()) | set(df["away_team_fbref"].unique())
        for team in all_teams:
            self.team_elos[team] = self.base_elo
        
        # 按时间排序比赛
        df_sorted = df.sort_values("jc_date")
        
        for _, row in df_sorted.iterrows():
            home_team = row["home_team_fbref"]
            away_team = row["away_team_fbref"]
            home_goals = row["home_goals"]
            away_goals = row["away_goals"]
            
            # 获取当前Elo
            home_elo = self.team_elos[home_team]
            away_elo = self.team_elos[away_team]
            
            # 计算预期胜率
            home_exp = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
            away_exp = 1 - home_exp
            
            # 实际结果（0=客胜，0.5=平，1=主胜）
            if home_goals > away_goals:
                home_actual = 1.0
                away_actual = 0.0
            elif home_goals == away_goals:
                home_actual = 0.5
                away_actual = 0.5
            else:
                home_actual = 0.0
                away_actual = 1.0
            
            # 调整Elo（考虑进球数加权）
            goal_diff = abs(home_goals - away_goals)
            k = self.k_factor * (1 + goal_diff / 2)  # 进球差越大，k值越大
            
            # 更新Elo
            self.team_elos[home_team] = home_elo + k * (home_actual - home_exp)
            self.team_elos[away_team] = away_elo + k * (away_actual - away_exp)
        
        logger.info(f"埃罗模型训练完成，共更新 {len(self.team_elos)} 支球队的Elo评分")
    
    def predict(self, home_team: str, away_team: str) -> dict:
        """预测胜平负概率"""
        home_elo = self.team_elos.get(home_team, self.base_elo)
        away_elo = self.team_elos.get(away_team, self.base_elo)
        
        # 计算预期胜率
        home_prob = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        away_prob = 1 - home_prob
        # 平局概率（经验值：Elo差值越小，平局概率越高）
        elo_diff = abs(home_elo - away_elo)
        draw_prob = max(0.1, 0.4 - (elo_diff / 200))  # Elo差越大，平局概率越低
        
        # 归一化概率
        total = home_prob + draw_prob + away_prob
        home_prob /= total
        draw_prob /= total
        away_prob /= total
        
        return {"home": home_prob, "draw": draw_prob, "away": away_prob}

class MonteCarloPredictor:
    """蒙特卡罗模拟预测"""
    def __init__(self, n_simulations: int = 10000):
        self.n_simulations = n_simulations
        self.poisson_model = PoissonPredictor()
    
    def fit(self, df: pd.DataFrame):
        """训练底层泊松模型"""
        self.poisson_model.fit(df)
    
    def simulate(self, league: str, home_team: str, away_team: str) -> dict:
        """蒙特卡罗模拟比赛结果"""
        # 获取泊松模型的预期进球数，添加保护措施
        avg_home, avg_away = self.poisson_model.league_avg_goals.get(league, (1.5, 1.2))
        home_attack = self.poisson_model.home_attack_strength.get((league, home_team), 1.0)
        home_defense = self.poisson_model.home_defense_strength.get((league, home_team), 1.0)
        away_attack = self.poisson_model.away_attack_strength.get((league, away_team), 1.0)
        away_defense = self.poisson_model.away_defense_strength.get((league, away_team), 1.0)
        
        # 确保lambda值为正数
        lambda_home = max(0.1, avg_home * home_attack * away_defense)
        lambda_away = max(0.1, avg_away * away_attack * home_defense)
        
        # 模拟n次比赛
        results = []
        for _ in range(self.n_simulations):
            home_goals = np.random.poisson(lambda_home)
            away_goals = np.random.poisson(lambda_away)
            
            if home_goals > away_goals:
                result = "home"
            elif home_goals == away_goals:
                result = "draw"
            else:
                result = "away"
            
            results.append({
                "home_goals": home_goals,
                "away_goals": away_goals,
                "result": result,
                "total_goals": home_goals + away_goals
            })
        
        # 统计结果
        results_df = pd.DataFrame(results)
        prob_home = len(results_df[results_df["result"] == "home"]) / self.n_simulations
        prob_draw = len(results_df[results_df["result"] == "draw"]) / self.n_simulations
        prob_away = len(results_df[results_df["result"] == "away"]) / self.n_simulations
        prob_over25 = len(results_df[results_df["total_goals"] > 2.5]) / self.n_simulations
        
        return {
            "1x2_prob": {"home": prob_home, "draw": prob_draw, "away": prob_away},
            "over25_prob": prob_over25,
            "avg_home_goals": results_df["home_goals"].mean(),
            "avg_away_goals": results_df["away_goals"].mean()
        }
