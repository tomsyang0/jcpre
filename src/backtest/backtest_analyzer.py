import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from config import BACKTEST_DIR

class BacktestAnalyzer:
    """策略回测分析"""
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.backtest_results = None
    
    def backtest_betting_strategy(self, df: pd.DataFrame, bet_size: float = 1.0, strategy: str = "value_bet") -> pd.DataFrame:
        """
        回测投注策略：
        - strategy: value_bet（价值投注）/ all_bet（全部投注）/ random_bet（随机投注）
        """
        df_backtest = df.copy()
        df_backtest["stake"] = bet_size  # 固定投注金额
        df_backtest["return"] = 0.0
        df_backtest["capital"] = 0.0
        
        capital = self.initial_capital
        
        for idx, row in df_backtest.iterrows():
            # 确定投注选项
            if strategy == "value_bet":
                # 只投注正EV的选项
                if not row["has_value_bet"]:
                    df_backtest.loc[idx, "capital"] = capital
                    continue
                
                # 获取推荐的投注选项
                rec = row["betting_recommendation"]
                if "主胜" in rec:
                    bet_result = "home"
                elif "平局" in rec:
                    bet_result = "draw"
                elif "客胜" in rec:
                    bet_result = "away"
                else:
                    df_backtest.loc[idx, "capital"] = capital
                    continue
            elif strategy == "all_bet":
                # 全部投注主胜
                bet_result = "home"
            elif strategy == "random_bet":
                # 随机投注
                bet_result = np.random.choice(["home", "draw", "away"])
            else:
                raise ValueError(f"不支持的策略：{strategy}")
            
            # 检查实际比赛结果
            actual_result = row["result_1x2"]
            actual_result_map = {"H": "home", "D": "draw", "A": "away"}
            actual_bet_result = actual_result_map.get(actual_result, "")
            
            # 计算收益
            odds = row[f"spf_sp_{bet_result}_close"]
            if bet_result == actual_bet_result:
                # 赢钱
                profit = (odds - 1) * bet_size * (1 - 0.02)  # 扣除佣金
                capital += profit
            else:
                # 输钱
                capital -= bet_size
            
            df_backtest.loc[idx, "return"] = profit if bet_result == actual_bet_result else -bet_size
            df_backtest.loc[idx, "capital"] = capital
        
        # 计算回测指标
        total_bets = len(df_backtest[df_backtest["stake"] > 0])
        winning_bets = len(df_backtest[df_backtest["return"] > 0])
        win_rate = winning_bets / total_bets if total_bets > 0 else 0
        total_profit = capital - self.initial_capital
        roi = (total_profit / self.initial_capital) * 100
        max_drawdown = self.calculate_max_drawdown(df_backtest["capital"])
        
        self.backtest_results = {
            "initial_capital": self.initial_capital,
            "final_capital": capital,
            "total_profit": total_profit,
            "roi": roi,
            "total_bets": total_bets,
            "winning_bets": winning_bets,
            "win_rate": win_rate,
            "max_drawdown": max_drawdown
        }
        
        logger.info(f"回测完成：")
        logger.info(f"初始资金：{self.initial_capital:.2f}，最终资金：{capital:.2f}")
        logger.info(f"总收益：{total_profit:.2f}，ROI：{roi:.2f}%")
        logger.info(f"总投注数：{total_bets}，胜率：{win_rate:.2f}%")
        logger.info(f"最大回撤：{max_drawdown:.2f}%")
        
        # 保存回测结果
        self.save_backtest_results(df_backtest, strategy)
        # 绘制资金曲线
        self.plot_capital_curve(df_backtest, strategy)
        
        return df_backtest
    
    def calculate_max_drawdown(self, capital_series: pd.Series) -> float:
        """计算最大回撤（百分比）"""
        peak = capital_series.expanding(min_periods=1).max()
        drawdown = (capital_series - peak) / peak * 100
        max_drawdown = drawdown.min()
        return max_drawdown
    
    def save_backtest_results(self, df: pd.DataFrame, strategy: str):
        """保存回测结果到文件"""
        # 保存详细结果
        df.to_csv(BACKTEST_DIR / f"backtest_{strategy}_detailed.csv", index=False, encoding="utf-8")
        
        # 保存回测指标
        results_df = pd.DataFrame([self.backtest_results])
        results_df.to_csv(BACKTEST_DIR / f"backtest_{strategy}_metrics.csv", index=False, encoding="utf-8")
        
        logger.info(f"回测结果已保存到：{BACKTEST_DIR}")
    
    def plot_capital_curve(self, df: pd.DataFrame, strategy: str):
        """绘制资金曲线"""
        plt.figure(figsize=(12, 6))
        
        # 检查是否有日期列，如果没有则使用索引
        if "jc_date" in df.columns:
            plt.plot(df["jc_date"], df["capital"], label=f"{strategy} strategy")
            plt.xlabel("Date")
        else:
            plt.plot(df.index, df["capital"], label=f"{strategy} strategy")
            plt.xlabel("Bet Number")
        
        plt.title(f"Capital Curve - {strategy} Strategy")
        plt.ylabel("Capital")
        plt.grid(True)
        plt.legend()
        plt.savefig(BACKTEST_DIR / f"capital_curve_{strategy}.png", dpi=300, bbox_inches="tight")
        plt.close()
