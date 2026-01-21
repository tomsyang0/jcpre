import pandas as pd
import numpy as np
from loguru import logger

class BettingAnalyzer:
    """期望价值(EV)和凯利指数分析"""
    def __init__(self, commission_rate: float = 0.02):
        self.commission_rate = commission_rate  # 投注佣金率
    
    def calculate_ev(self, win_prob: float, odds: float, stake: float = 1.0) -> float:
        """
        计算期望价值(EV)
        EV = (win_prob * (odds - 1) - (1 - win_prob)) * stake * (1 - commission_rate)
        """
        ev = (win_prob * (odds - 1) - (1 - win_prob)) * stake * (1 - self.commission_rate)
        return ev
    
    def calculate_kelley(self, win_prob: float, odds: float) -> float:
        """
        计算凯利指数（最优投注比例）
        Kelly % = (bp - q) / b
        其中：b = 赔率-1, p = 赢的概率, q = 输的概率(1-p)
        """
        b = odds - 1
        q = 1 - win_prob
        kelley = (win_prob * b - q) / b
        
        # 限制凯利指数在0-1之间
        kelley = max(0.0, min(1.0, kelley))
        return kelley
    
    def analyze_value_bets(self, df: pd.DataFrame, pred_prob: dict) -> pd.DataFrame:
        """
        分析价值投注：
        1. 计算每个投注选项的EV和凯利指数
        2. 筛选正EV的投注
        3. 生成投注推荐
        """
        df_betting = df.copy()
        
        # 胜平负概率（模型预测）
        df_betting["pred_home_prob"] = pred_prob["home"]
        df_betting["pred_draw_prob"] = pred_prob["draw"]
        df_betting["pred_away_prob"] = pred_prob["away"]
        
        # 计算胜平负的EV和凯利指数
        for result in ["home", "draw", "away"]:
            # 赔率列
            odds_col = f"spf_sp_{result}_close"
            if odds_col not in df_betting.columns:
                continue
            
            # 计算EV
            df_betting[f"ev_{result}"] = df_betting.apply(
                lambda row: self.calculate_ev(row[f"pred_{result}_prob"], row[odds_col]),
                axis=1
            )
            
            # 计算凯利指数
            df_betting[f"kelley_{result}"] = df_betting.apply(
                lambda row: self.calculate_kelley(row[f"pred_{result}_prob"], row[odds_col]),
                axis=1
            )
        
        # 筛选价值投注（正EV）
        df_betting["has_value_bet"] = (
            (df_betting["ev_home"] > 0) | 
            (df_betting["ev_draw"] > 0) | 
            (df_betting["ev_away"] > 0)
        )
        
        # 生成投注推荐
        def get_recommendation(row):
            if not row["has_value_bet"]:
                return "无价值投注"
            
            # 选择EV最大的选项
            ev_vals = [row["ev_home"], row["ev_draw"], row["ev_away"]]
            max_ev_idx = np.argmax(ev_vals)
            result_map = {0: "home", 1: "draw", 2: "away"}
            max_ev_key = result_map[max_ev_idx]
            
            # 中文映射
            cn_map = {"home": "主胜", "draw": "平局", "away": "客胜"}
            max_ev_cn = cn_map[max_ev_key]
            
            # 获取对应的EV和凯利指数
            ev_value = row[f"ev_{max_ev_key}"]
            kelley_value = row[f"kelley_{max_ev_key}"]
            
            return f"{max_ev_cn} (EV:{ev_value:.4f}, 凯利:{kelley_value:.4f})"
        
        df_betting["betting_recommendation"] = df_betting.apply(get_recommendation, axis=1)
        
        logger.info(f"价值投注分析完成，共找到 {df_betting['has_value_bet'].sum()} 场有价值的比赛")
        return df_betting