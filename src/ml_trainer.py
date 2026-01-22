import lightgbm as lgb
import joblib

class MLTrainer:
    def train(self, X, y):
        model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31, 
                                  reg_alpha=0.1, reg_lambda=0.1, importance_type='gain')
        model.fit(X, y)
        joblib.dump(model, 'models/ml_model.pkl')