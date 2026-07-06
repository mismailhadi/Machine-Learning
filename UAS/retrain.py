import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
import warnings
warnings.filterwarnings('ignore')

print("Memulai proses pembuatan ulang model...")

# 1. Load Data
df = pd.read_csv('arrhythmia.data', header=None)
df.replace('?', np.nan, inplace=True)
df_clean = df.drop(columns=[13]).drop_duplicates()

X = df_clean.drop(columns=[279])
y = df_clean[279]

# 2. Preprocessing
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

selector = VarianceThreshold(threshold=0.0)
X_selected = selector.fit_transform(X_scaled)

# 3. Handle Imbalance
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X_selected, y)

# 4. Latih Model SVM (Langsung pakai parameter terbaik dari Soal 4)
model = SVC(C=10, gamma='auto', kernel='rbf', probability=True, random_state=42)
model.fit(X_res, y_res)

# 5. Timpa file .joblib lama dengan yang baru (versi lokal)
joblib.dump(model, 'best_arrhythmia_model.joblib')
joblib.dump(imputer, 'imputer.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(selector, 'selector.joblib')

print("✅ SUKSES! File .joblib baru telah dibuat dan cocok dengan versi Python lokalmu.")