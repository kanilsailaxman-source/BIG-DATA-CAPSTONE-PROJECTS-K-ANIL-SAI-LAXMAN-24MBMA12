
import pandas as pd, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
PROJ = Path(__file__).resolve().parents[0]
df = pd.read_csv(PROJ/'data'/'dataset.csv')
scenarios = {'base':{'PD_mult':1.0,'LGD_add':0.0}, 'mild_recession':{'PD_mult':1.5,'LGD_add':0.05}, 'severe_recession':{'PD_mult':2.5,'LGD_add':0.12}}
n_sims = 2000; out = PROJ/'outputs'; out.mkdir(exist_ok=True)
results = {}
for name, sc in scenarios.items():
    losses = []
    for _ in range(n_sims):
        PD = np.clip(df['PD_base']*sc['PD_mult'],0,1)
        LGD = np.clip(df['LGD']+sc['LGD_add'],0,1)
        default = np.random.rand(len(df)) < PD
        loss = (df['EAD']*LGD*default).sum()
        losses.append(loss)
    arr = np.array(losses); results[name]=arr
    plt.figure(); plt.hist(arr, bins=40); plt.tight_layout(); plt.savefig(out/f'loss_dist_{name}.png', dpi=150)
summary = [{'scenario':name, 'EL_mean':arr.mean(), 'P95':np.percentile(arr,95), 'P99':np.percentile(arr,99)} for name,arr in results.items()]
pd.DataFrame(summary).to_csv(out/'summary.csv', index=False)
