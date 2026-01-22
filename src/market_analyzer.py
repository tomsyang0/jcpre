class MarketAnalyzer:
    def __init__(self):
        self.jc_payout = 0.88 # 竞彩平均返还率

    def get_value_report(self, prob, odds, betting_single):
        # 计算EV（期望价值）
        ev = (prob * odds) - 1
        
        # 判定级别：竞彩环境下，EV > 0 即具备博弈价值
        level = "观望"
        if ev > 0.05: level = "建议串关"
        if ev > 0.10: level = "高价值"
        if betting_single == 1 and ev > 0.02: level = "重点单场"
        
        return ev, level