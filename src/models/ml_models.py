import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from loguru import logger
import joblib
from config import MODEL_DIR, RANDOM_STATE, TEST_SIZE

class MLModelTrainer:
    """机器学习模型训练器（胜平负/让球胜平负/大小球预测）"""
    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()  # 新增：标签编码器
        self.model_path = MODEL_DIR / f"{model_type}_predictor.pkl"
        # 定义标签映射（方便解码）
        self.label_mapping = {'H': 0, 'D': 1, 'A': 2}
        self.reverse_label_mapping = {0: 'H', 1: 'D', 2: 'A'}
    
    def build_model(self):
        """构建指定类型的模型"""
        if self.model_type == "logistic":
            self.model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
        elif self.model_type == "rf":
            self.model = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100)
        elif self.model_type == "gbdt":
            self.model = GradientBoostingClassifier(random_state=RANDOM_STATE, n_estimators=100)
        elif self.model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                random_state=RANDOM_STATE, 
                n_estimators=100, 
                use_label_encoder=False, 
                eval_metric="mlogloss",
                num_class=3  # 明确指定3分类
            )
        elif self.model_type == "lightgbm":
            self.model = lgb.LGBMClassifier(
                random_state=RANDOM_STATE, 
                n_estimators=100,
                num_class=3
            )
        else:
            raise ValueError(f"不支持的模型类型：{self.model_type}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series, cv: int = 5):
        """训练模型并评估（新增标签编码）"""
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        
        # 标签编码（字符→数值）
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # 特征标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 构建并训练模型
        self.build_model()
        self.model.fit(X_train_scaled, y_train_encoded)
        
        # 模型评估（解码回字符型方便查看）
        y_pred_encoded = self.model.predict(X_test_scaled)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train_encoded, cv=cv)
        
        logger.info(f"模型准确率：{accuracy:.4f}")
        logger.info(f"交叉验证平均分：{cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        logger.info(f"分类报告：\n{classification_report(y_test, y_pred)}")
        
        # 保存模型（包含标签编码器）
        self.save_model()
        return accuracy
    
    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """预测结果和概率（返回字符型标签）"""
        if self.model is None:
            self.load_model()
        
        X_scaled = self.scaler.transform(X)
        predictions_encoded = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # 解码为字符型标签（H/D/A）
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        
        return predictions, probabilities
    
    def save_model(self):
        """保存模型到文件（包含标签编码器）"""
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,  # 保存编码器
            "model_type": self.model_type,
            "label_mapping": self.label_mapping,
            "reverse_label_mapping": self.reverse_label_mapping
        }, self.model_path)
        logger.info(f"模型已保存到：{self.model_path}")
    
    def load_model(self):
        """从文件加载模型（包含标签编码器）"""
        if self.model_path.exists():
            data = joblib.load(self.model_path)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.label_encoder = data["label_encoder"]  # 加载编码器
            self.model_type = data["model_type"]
            self.label_mapping = data.get("label_mapping", {'H': 0, 'D': 1, 'A': 2})
            self.reverse_label_mapping = data.get("reverse_label_mapping", {0: 'H', 1: 'D', 2: 'A'})
            logger.info(f"模型已从 {self.model_path} 加载")
        else:
            raise FileNotFoundError(f"模型文件不存在：{self.model_path}")
    
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """获取特征重要性"""
        if self.model is None:
            self.load_model()
        
        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
            importance_df = pd.DataFrame({
                "feature": feature_names,
                "importance": importance
            }).sort_values("importance", ascending=False)
            return importance_df
        else:
            logger.warning("当前模型不支持特征重要性分析")
            return pd.DataFrame()
