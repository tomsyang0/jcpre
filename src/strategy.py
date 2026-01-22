import pandas as pd

class StrategyManager:
    def __init__(self, bankroll=10000, kelly_fraction=0.1):
        self.bankroll = bankroll         # 总本金
        self.kelly_fraction = kelly_fraction # 凯利系数（稳健型设为0.1）

    def analyze_all_options(self, row, final_probs):
        """扫描胜平负，寻找最优EV及注码比例"""
        # H:胜, D:平, A:负
        opts = ['H', 'D', 'A']
        odds = [row['spf_sp_home_now'], row['spf_sp_draw_now'], row['spf_sp_away_now']]
        
        results = []
        for i in range(3):
            prob = final_probs[i]
            odd = odds[i]
            if pd.isna(odd) or odd <= 1.0: continue
            
            ev = (prob * odd) - 1
            # 凯利公式: f = (bp - q) / b
            b = odd - 1
            f = (b * prob - (1 - prob)) / b if b > 0 else 0
            kelly_stake = max(0, f * self.kelly_fraction)
            
            results.append({'opt': opts[i], 'ev': ev, 'kelly': kelly_stake, 'prob': prob, 'odd': odd})
        
        # 筛选最优选项
        if not results: return None
        best = max(results, key=lambda x: x['ev'])
        return best