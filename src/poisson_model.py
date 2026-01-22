import numpy as np
from scipy.stats import poisson

class PoissonModel:
    def calculate_prob(self, row):
        # 2026 节奏校准
        h_lambda = (row['home_xG'] if row['home_xG'] > 0 else 1.2) * (1 + row['tempo_diff']*0.05)
        a_lambda = (row['away_xG'] if row['away_xG'] > 0 else 1.0)
        h_probs = poisson.pmf(range(7), h_lambda)
        a_probs = poisson.pmf(range(7), a_lambda)
        m = np.outer(h_probs, a_probs)
        return [np.sum(np.tril(m, -1)), np.sum(np.diag(m)), np.sum(np.triu(m, 1))]

