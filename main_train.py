import pandas as pd
import torch
import joblib
from src.data_processor import DataProcessor
from src.ml_trainer import MLTrainer
from src.dl_trainer import FootballNet

def main():
    dp = DataProcessor()
    df = pd.read_csv('datasets/jc_fbref_bonus_support_feature_2025-01-01_to_2026-01-21.csv')
    X, y = dp.process_for_train(df)
    
    # ML 训练
    MLTrainer().train(X, y)
    
    # DL 训练 (残差网络)
    model = FootballNet(X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    X_t, y_t = torch.FloatTensor(X.values), torch.LongTensor(y.values)
    
    for epoch in range(150):
        optimizer.zero_grad()
        loss = torch.nn.CrossEntropyLoss()(model(X_t), y_t)
        loss.backward()
        optimizer.step()
        
    torch.save(model.state_dict(), 'models/dl_model.pth')
    print(f"2026赛季模型训练完成，样本量: {len(X)}")

if __name__ == "__main__": main()
