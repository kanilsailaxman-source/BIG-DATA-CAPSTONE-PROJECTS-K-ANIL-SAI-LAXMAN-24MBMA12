
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from pathlib import Path
PROJ = Path(__file__).resolve().parents[0]
df = pd.read_csv(PROJ/'data'/'dataset.csv', parse_dates=['date'])
df['label'] = df['sentiment'].astype(str)
train, test = train_test_split(df, test_size=0.25, random_state=42, stratify=df['label'])
pipe = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression(max_iter=300))])
pipe.fit(train['review_text'], train['label'])
preds = pipe.predict(test['review_text'])
print('Accuracy:', accuracy_score(test['label'], preds))
df['week'] = pd.to_datetime(df['date']).dt.to_period('W').apply(lambda r: r.start_time)
trend = df.groupby(['week','department'])['rating'].mean().reset_index()
pivot = trend.pivot(index='week', columns='department', values='rating').fillna(method='ffill')
out = PROJ/'outputs'; out.mkdir(exist_ok=True)
ax = pivot.plot(figsize=(10,4)); plt.tight_layout(); plt.savefig(out/'sentiment_trend.png', dpi=150)
