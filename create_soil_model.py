# create_soil_model.py -- simple toy classifier to generate models/soil_model.joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os

# synthetic training data (toy)
rng = np.random.RandomState(0)
n = 1000
N = rng.normal(loc=40, scale=20, size=n).clip(0,200)
P = rng.normal(loc=20, scale=10, size=n).clip(0,100)
K = rng.normal(loc=150, scale=80, size=n).clip(0,400)
ph = rng.normal(loc=6.5, scale=0.8, size=n).clip(3.5,9)
moist = rng.normal(loc=20, scale=5, size=n).clip(0,60)

# label by simple rule used earlier
def label_row(nv, pv, kv):
    score = min(nv/40.0,2.0)*0.4 + min(pv/20.0,2.0)*0.3 + min(kv/150.0,2.0)*0.3
    if score >= 1.2:
        return "High"
    elif score >= 0.7:
        return "Medium"
    else:
        return "Low"

y = [label_row(N[i], P[i], K[i]) for i in range(n)]
X = pd.DataFrame({"N":N, "P":P, "K":K, "ph":ph, "moisture":moist})

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(n_estimators=50, random_state=0))
])

pipe.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(pipe, "models/soil_model.joblib")
print("Saved toy model to models/soil_model.joblib")
