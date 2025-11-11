
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
PROJ = Path(__file__).resolve().parents[0]
df = pd.read_csv(PROJ/'data'/'dataset.csv', parse_dates=['timestamp'])
y = df['label_fraud']
X = df.drop(columns=['label_fraud','tx_id','timestamp'])
cat_cols = ['merchant_type','country','device']
num_cols = [c for c in X.columns if c not in cat_cols]
pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols), ('num','passthrough', num_cols)])
pipe = Pipeline([('pre', pre), ('clf', LogisticRegression(max_iter=300))])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
pipe.fit(X_train, y_train)
proba = pipe.predict_proba(X_test)[:,1]
preds = (proba>=0.5).astype(int)
print('ROC AUC:', roc_auc_score(y_test, proba))
print(classification_report(y_test, preds))
out = PROJ/'outputs'; out.mkdir(exist_ok=True)
ths = np.linspace(0,1,200); tpr=[]; fpr=[]
from sklearn.metrics import confusion_matrix
for t in ths:
    p = (proba>=t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, p).ravel()
    tpr.append(tp/(tp+fn) if (tp+fn)>0 else 0)
    fpr.append(fp/(fp+tn) if (fp+tn)>0 else 0)
plt.figure(); plt.plot(fpr,tpr); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC (threshold sweep)'); plt.tight_layout(); plt.savefig(out/'roc_curve.png', dpi=150)
