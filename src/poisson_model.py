import numpy as np
import pandas as pd
from scipy.stats import poisson

class PoissonModel:
    def calculate_prob(self, row):
        # 使用与calculate_scoreline_probs相同的归一化逻辑
        def normalize_xg(xg, is_home=True):
            if pd.isna(xg) or xg <= 0:
                return 1.4 if is_home else 1.0  # 主场默认值更高
            
            # 将xg从12-48范围映射到更广的范围，增加差异
            # 主场: 1.2-2.8, 客场: 0.8-2.2
            if is_home:
                base = 1.2
                scale = 1.6
            else:
                base = 0.8
                scale = 1.4
            
            xg_normalized = base + scale * (xg - 12) / 36 if xg > 12 else base
            # 确保在合理范围内
            return max(0.6, min(3.5, xg_normalized))
        
        # 计算进球期望值，增加主场优势
        home_xg_norm = normalize_xg(row.get('home_xG'), is_home=True)
        away_xg_norm = normalize_xg(row.get('away_xG'), is_home=False)
        
        # 考虑主客场节奏差异和其他因素
        tempo_factor = 1 + (row.get('tempo_diff', 0) * 0.03 if not pd.isna(row.get('tempo_diff')) else 0)
        
        # 额外的主场优势因子
        home_advantage = 1.15  # 15%的主场优势
        
        h_lambda = home_xg_norm * tempo_factor * home_advantage
        a_lambda = away_xg_norm
        
        # 确保lambda值有最小差异，避免太多平局
        if abs(h_lambda - a_lambda) < 0.3:
            # 如果两队实力太接近，给主场队一点优势
            h_lambda += 0.2
            a_lambda = max(0.7, a_lambda - 0.1)
        
        h_probs = poisson.pmf(range(7), h_lambda)
        a_probs = poisson.pmf(range(7), a_lambda)
        m = np.outer(h_probs, a_probs)
        return [np.sum(np.tril(m, -1)), np.sum(np.diag(m)), np.sum(np.triu(m, 1))]
    
    def calculate_scoreline_probs(self, row, top_n=3):
        """计算最可能的比分及其概率"""
        # 调整归一化函数，增加lambda值的差异，减少1-1平局概率
        def normalize_xg(xg, is_home=True):
            if pd.isna(xg) or xg <= 0:
                return 1.4 if is_home else 1.0  # 主场默认值更高
            
            # 将xg从12-48范围映射到更广的范围，增加差异
            # 主场: 1.2-2.8, 客场: 0.8-2.2
            if is_home:
                base = 1.2
                scale = 1.6
            else:
                base = 0.8
                scale = 1.4
            
            xg_normalized = base + scale * (xg - 12) / 36 if xg > 12 else base
            # 确保在合理范围内
            return max(0.6, min(3.5, xg_normalized))
        
        # 计算进球期望值，增加主场优势
        home_xg_norm = normalize_xg(row.get('home_xG'), is_home=True)
        away_xg_norm = normalize_xg(row.get('away_xG'), is_home=False)
        
        # 考虑主客场节奏差异和其他因素
        tempo_factor = 1 + (row.get('tempo_diff', 0) * 0.03 if not pd.isna(row.get('tempo_diff')) else 0)
        
        # 额外的主场优势因子
        home_advantage = 1.15  # 15%的主场优势
        
        h_lambda = home_xg_norm * tempo_factor * home_advantage
        a_lambda = away_xg_norm
        
        # 确保lambda值有足够差异，减少平局概率
        min_lambda_diff = 0.5  # 增加最小差异
        if abs(h_lambda - a_lambda) < min_lambda_diff:
            # 如果两队实力太接近，增加差异
            diff_needed = min_lambda_diff - abs(h_lambda - a_lambda)
            h_lambda += diff_needed * 0.7
            a_lambda = max(0.6, a_lambda - diff_needed * 0.3)
        
        # 调整lambda值，让强队更强，弱队更弱
        # 使用非线性变换增加差异
        h_lambda = h_lambda ** 1.1  # 稍微增加强队的lambda
        a_lambda = max(0.5, a_lambda ** 0.95)  # 稍微减少弱队的lambda
        
        # 计算0-5个进球的概率（减少最大进球数以增加低比分概率）
        max_goals = 5
        h_probs = poisson.pmf(range(max_goals + 1), h_lambda)
        a_probs = poisson.pmf(range(max_goals + 1), a_lambda)
        
        # 对低比分（0-2球）给予额外权重，减少1-1平局
        # 创建加权概率矩阵
        score_probs = np.outer(h_probs, a_probs)
        
        # 应用比分权重：降低1-1的权重，提高1-0、2-0、2-1等的权重
        weight_matrix = np.ones((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                if i == j:  # 平局
                    if i == 1:  # 1-1平局
                        weight_matrix[i, j] = 0.7  # 降低30%权重
                    else:
                        weight_matrix[i, j] = 0.9  # 降低10%权重
                elif abs(i - j) == 1:  # 一球之差
                    weight_matrix[i, j] = 1.1  # 增加10%权重
                elif abs(i - j) >= 2:  # 两球或以上之差
                    weight_matrix[i, j] = 1.05  # 增加5%权重
        
        score_probs = score_probs * weight_matrix
        score_probs = score_probs / score_probs.sum()  # 重新归一化
        
        # 获取所有可能的比分及其概率
        scorelines = []
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                prob = score_probs[i, j]
                if prob > 0.001:  # 提高阈值，只显示概率较高的比分
                    scorelines.append((f"{i}-{j}", prob))
        
        # 按概率降序排序
        scorelines.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前top_n个最可能的比分
        top_n_display = min(top_n, len(scorelines))
        top_scorelines = scorelines[:top_n_display] if scorelines else [("1-0", 0.15), ("2-1", 0.12), ("2-0", 0.10)]
        
        # 格式化输出
        formatted = []
        for scoreline, prob in top_scorelines:
            formatted.append(f"{scoreline}: {prob:.2%}")
        
        return "; ".join(formatted), top_scorelines
