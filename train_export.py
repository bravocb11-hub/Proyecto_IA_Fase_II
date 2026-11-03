"""Reentrena y exporta el baseline de la Fase I con el mismo preprocesamiento."""
from pathlib import Path
import json
import kagglehub
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

RANDOM_STATE = 42
path = Path(kagglehub.dataset_download("tobiasbueck/multilingual-customer-support-tickets"))
files = sorted([p for p in path.rglob('*') if p.suffix.lower() in {'.csv','.xlsx','.xls'}])
if not files: raise FileNotFoundError('No hay CSV o Excel en el dataset descargado')
f = files[0]
df = pd.read_csv(f) if f.suffix.lower()=='.csv' else pd.read_excel(f)
norm = {c.lower().strip().replace(' ','_'):c for c in df.columns}
def pick(names): return next((norm[n] for n in names if n in norm),None)
subject_col=pick(['subject','ticket_subject'])
body_col=pick(['body','ticket_description','description','text'])
queue_col=pick(['queue','department','ticket_type'])
if not body_col or not queue_col: raise ValueError(f'Columnas requeridas ausentes: {list(df.columns)}')
body=df[body_col].fillna('').astype(str)
text=(df[subject_col].fillna('').astype(str)+' '+body).str.strip() if subject_col else body.str.strip()
data=pd.DataFrame({'text':text,'queue':df[queue_col]}).dropna(subset=['queue'])
data=data[data.text.str.len()>0]
counts=data.queue.astype(str).value_counts()
data=data[data.queue.astype(str).isin(counts[counts>=2].index)]
X_train,X_test,y_train,y_test=train_test_split(data.text,data.queue.astype(str),test_size=.20,random_state=RANDOM_STATE,stratify=data.queue.astype(str))
pipe=Pipeline([('tfidf',TfidfVectorizer(lowercase=True,ngram_range=(1,2),min_df=2,max_features=30000,sublinear_tf=True)),('clf',LogisticRegression(max_iter=1500,class_weight='balanced',random_state=RANDOM_STATE))])
pipe.fit(X_train,y_train)
pred=pipe.predict(X_test)
metrics={'accuracy':float(accuracy_score(y_test,pred)),'f1_macro':float(f1_score(y_test,pred,average='macro',zero_division=0)),'n_train':len(X_train),'n_test':len(X_test),'dataset_file':f.name}
Path('models').mkdir(exist_ok=True)
joblib.dump(pipe,'models/ticket_pipeline.joblib')
Path('models/metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(metrics,indent=2,ensure_ascii=False))
