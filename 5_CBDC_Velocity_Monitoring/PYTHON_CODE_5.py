
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
PROJ = Path(__file__).resolve().parents[0]
df = pd.read_csv(PROJ/'data'/'dataset.csv', parse_dates=['timestamp'])
df['hour'] = df['timestamp'].dt.floor('H')
hourly = df.groupby('hour')['amount'].sum().reset_index()
hourly['roll_mean'] = hourly['amount'].rolling(24, min_periods=6).mean()
hourly['roll_std'] = hourly['amount'].rolling(24, min_periods=6).std()
hourly['z'] = (hourly['amount'] - hourly['roll_mean']) / hourly['roll_std']
hourly['peak'] = (hourly['z'] > 2).astype(int)
out = PROJ/'outputs'; out.mkdir(exist_ok=True)
plt.figure(figsize=(9,4)); plt.plot(hourly['hour'], hourly['amount']); plt.scatter(hourly.loc[hourly['peak']==1,'hour'], hourly.loc[hourly['peak']==1,'amount']); plt.tight_layout(); plt.savefig(out/'cbdc_hourly.png', dpi=150)
pivot = df.pivot_table(index=df['timestamp'].dt.date, columns='region', values='amount', aggfunc='sum').fillna(0)
pivot.to_csv(out/'region_day_matrix.csv')
