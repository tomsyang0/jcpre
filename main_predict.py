import pandas as pd
import numpy as np
import torch
import glob, os, joblib
from datetime import datetime
from src.data_processor import DataProcessor
from src.dl_trainer import FootballNet
from src.ml_trainer import MLTrainer
from src.poisson_model import PoissonModel
from src.strategy import StrategyManager

def main():
    dp, pm, ml = DataProcessor(), PoissonModel(), MLTrainer()
    sm = StrategyManager(bankroll=10000, kelly_fraction=0.1) # 1万本金
    
    # 1. 加载模型与数据
    files = glob.glob('predict_data/jc_today_unplayed_for_predict_*.csv')
    df_raw = pd.read_csv(max(files, key=os.path.getctime))
    X_scaled, df_clean = dp.process_for_predict(df_raw)
    
    ml_model = joblib.load('models/ml_model.pkl')
    dl_model = FootballNet(X_scaled.shape[1])
    dl_model.load_state_dict(torch.load('models/dl_model.pth'))
    dl_model.eval()

    # 2. 推理与评估
    results = []
    with torch.no_grad():
        probs_ml = ml_model.predict_proba(X_scaled)
        probs_dl = torch.softmax(dl_model(torch.FloatTensor(X_scaled.values)), dim=1).numpy()

    for i, row in df_clean.iterrows():
        p_poisson = np.array(pm.calculate_prob(row))
        # 三模融合概率 [H, D, A]
        final_probs = probs_ml[i]*0.5 + probs_dl[i]*0.3 + p_poisson*0.2
        
        # 全向价值扫描
        decision = sm.analyze_all_options(row, final_probs)
        if decision and decision['ev'] > 0.02: # 门槛：2% 优势
            results.append({
                '场次': row['match_num'],
                '对阵': f"{row['home_team_cn']} vs {row['away_team_cn']}",
                '类型': '单场' if row.get('bettingSingle') == 1 else '串关',
                '推荐方向': {'H':'胜','D':'平','A':'负'}[decision['opt']],
                '预测概率': f"{decision['prob']:.1%}",
                '竞彩赔率': decision['odd'],
                'EV': f"{decision['ev']:.2%}",
                '建议投注': f"{sm.bankroll * decision['kelly']:.0f}元",
                '本金占比': f"{decision['kelly']:.2%}"
            })

    # 3. 输出导出
    res_df = pd.DataFrame(results)
    output_path = f"outputs/betting_plan_{datetime.now().strftime('%Y%m%d')}.csv"
    res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n--- 2026-01-22 盈利性投注计划 ---")
    print(res_df.sort_values(by='EV', ascending=False).to_string(index=False))
    print(f"\n[提示] 完整注码清单已存至: {output_path}")

if __name__ == "__main__": main()
