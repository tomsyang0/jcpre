import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_cols = [
            'hcp_p_home_now', 'hcp_p_draw_now', 'hcp_p_away_now',
            'home_win_rate', 'away_win_rate', 'home_away_goal_diff',
            'home_xG', 'away_xG', 'home_def_strength', 'away_def_strength',
            'home_attack_v_away_def', 'away_attack_v_home_def',
            'home_h_bias', 'away_a_bias', 'tempo_diff'
        ]

    def clean_and_eng(self, df):
        # 计算核心衍生特征
        df['home_attack_v_away_def'] = df['home_off_buildup'] - df['away_def_strength']
        df['away_attack_v_home_def'] = df['away_off_buildup'] - df['home_def_strength']
        df['home_h_bias'] = df['home_Home.xG'] - df['home_xG']
        df['away_a_bias'] = df['away_Away.xG'] - df['away_xG']
        
        for col in self.feature_cols:
            if col not in df.columns: df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        return df

    def process_for_train(self, df):
        df = self.clean_and_eng(df)
        df['target'] = df['result_1x2'].astype(str).str.strip().map({'H': 0, 'D': 1, 'A': 2})
        train_df = df.dropna(subset=['target']).copy()
        X = self.scaler.fit_transform(train_df[self.feature_cols])
        joblib.dump(self.scaler, 'models/scaler.pkl')
        # 转换为DataFrame以保留特征名称
        X_df = pd.DataFrame(X, columns=self.feature_cols)
        return X_df, train_df['target'].astype(int)

    def process_for_predict(self, df):
        df = self.clean_and_eng(df)
        scaler = joblib.load('models/scaler.pkl')
        X_scaled = scaler.transform(df[self.feature_cols])
        # 转换为DataFrame以保留特征名称，避免LightGBM警告
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_cols)
        return X_scaled_df, df
