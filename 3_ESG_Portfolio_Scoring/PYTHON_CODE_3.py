
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
PROJ = Path(__file__).resolve().parents[0]
df = pd.read_csv(PROJ/'data'/'dataset.csv')
df['ESG_score'] = 0.4*df['E_score'] + 0.3*df['S_score'] + 0.3*df['G_score']
asset_scores = df.groupby(['ticker','sector']).apply(lambda g: (g['ESG_score']*g['weight_norm']).sum()).reset_index(name='weighted_score')
out = PROJ/'outputs'; out.mkdir(exist_ok=True)
asset_scores.to_csv(out/'asset_scores.csv', index=False)
portfolio_score = (df['ESG_score']*df['weight_norm']).sum()
with open(out/'portfolio_score.txt','w') as f: f.write(f'Portfolio ESG Weighted Score: {portfolio_score:.2f}')
sector_avg = asset_scores.groupby('sector')['weighted_score'].mean().sort_values()
plt.figure(figsize=(8,4)); sector_avg.plot(kind='bar'); plt.tight_layout(); plt.savefig(out/'sector_esg.png', dpi=150)
