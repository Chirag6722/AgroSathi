import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "crop_model.joblib")

def build_synthetic_dataset():
    rows = []
    def add(soil, season, temp, ph, rain, crop):
        rows.append([soil, season, temp, ph, rain, crop])

    # base examples (expand as needed)
    add('loamy','monsoon',27,6.5,1200,'Rice')
    add('loamy','monsoon',28,6.8,900,'Maize')
    add('black','summer',30,7.0,500,'Cotton')
    add('sandy','summer',33,6.0,400,'Millet')
    add('clayey','winter',18,6.8,700,'Wheat')
    add('clayey','monsoon',25,6.0,900,'Paddy')
    add('red','summer',29,5.8,350,'Peanuts')
    add('loamy','winter',20,6.5,300,'Lentils')
    add('black','monsoon',26,7.2,1100,'Sugarcane')
    add('sandy','summer',32,6.5,300,'Groundnut')
    add('loamy','summer',30,6.8,450,'Sorghum')
    add('red','monsoon',28,6.0,600,'Pulses')
    add('black','summer',29,7.1,450,'Soybean')

    # add jittered copies to enlarge dataset
    base_len = len(rows)
    for _ in range(200):
        base = rows[np.random.randint(base_len)]
        soil, season, temp, ph, rain, crop = base
        rows.append([
            soil, season,
            max(0, float(temp) + np.random.randn()*2.0),
            round(max(3.0, min(9.0, float(ph) + np.random.randn()*0.25)), 1),
            max(0, int(float(rain) + np.random.randn()*60)),
            crop
        ])

    df = pd.DataFrame(rows, columns=['soil','season','temp','ph','rainfall','crop'])
    return df

def train_and_save():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = build_synthetic_dataset()
    X = df[['soil','season','temp','ph','rainfall']]
    y = df['crop']

    cat_cols = ['soil','season']
    num_cols = ['temp','ph','rainfall']

    preproc = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols)
    ])

    clf = Pipeline([
        ('pre', preproc),
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.12, random_state=42)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(f"Model trained. Test accuracy (synthetic): {score:.3f}")

    joblib.dump(clf, MODEL_PATH)
    print(f"Saved demo model to {MODEL_PATH}")

if __name__ == "__main__":
    train_and_save()
# train_crop_model.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "crop_model.joblib")

RANDOM_STATE = 42

def build_rich_synthetic_dataset(n_jitter=2300, random_state=RANDOM_STATE):
    """
    Build a larger synthetic dataset for crop recommendations.
    - n_jitter: how many jittered rows to generate in addition to base examples.
    Returns a pandas DataFrame with columns:
      ['soil','season','temp','ph','rainfall','crop']
    """
    np.random.seed(random_state)

    # Base soil types, seasons and some representative crops.
    soils = ['loamy', 'clayey', 'sandy', 'black', 'red', 'peaty', 'chalky']
    seasons = ['monsoon', 'summer', 'winter', 'spring', 'autumn']

    # A mapping of plausible crops for (soil, season) combos.
    # This is illustrative — adjust to your region/domain knowledge.
    crop_map = {
        ('loamy','monsoon'):    ['Rice','Maize','Sugarcane','Pulses'],
        ('loamy','summer'):     ['Sorghum','Maize','Groundnut','Millet'],
        ('loamy','winter'):     ['Wheat','Lentils','Barley'],
        ('loamy','spring'):     ['Vegetables','Maize'],
        ('loamy','autumn'):     ['Cotton','Soybean'],

        ('clayey','monsoon'):   ['Paddy','Sugarcane','Jute'],
        ('clayey','summer'):    ['Cotton','Millet','Peanuts'],
        ('clayey','winter'):    ['Wheat','Rape','Mustard'],
        ('clayey','spring'):    ['Vegetables','Potato'],
        ('clayey','autumn'):    ['Soybean','Pulses'],

        ('sandy','monsoon'):    ['Pearl millet','Groundnut','Pulses'],
        ('sandy','summer'):     ['Millet','Sorghum','Groundnut'],
        ('sandy','winter'):     ['Barley','Peas'],
        ('sandy','spring'):     ['Vegetables','Onion'],
        ('sandy','autumn'):     ['Sesame','Groundnut'],

        ('black','monsoon'):    ['Sugarcane','Rice','Soybean'],
        ('black','summer'):     ['Soybean','Cotton','Sunflower'],
        ('black','winter'):     ['Wheat','Gram'],
        ('black','spring'):     ['Vegetables','Maize'],
        ('black','autumn'):     ['Cotton','Pulses'],

        ('red','monsoon'):      ['Pulses','Maize'],
        ('red','summer'):       ['Groundnut','Sorghum','Cotton'],
        ('red','winter'):       ['Wheat','Mustard'],
        ('red','spring'):       ['Vegetables','Millets'],
        ('red','autumn'):       ['Pulses','Soybean'],

        ('peaty','monsoon'):    ['Rice','Sugarcane'],
        ('peaty','summer'):     ['Sugarcane','Vegetables'],
        ('peaty','winter'):     ['Wheat','Vegetables'],
        ('peaty','spring'):     ['Vegetables','Rice'],
        ('peaty','autumn'):     ['Rice','Pulses'],

        ('chalky','monsoon'):   ['Maize','Pulses'],
        ('chalky','summer'):    ['Sorghum','Groundnut'],
        ('chalky','winter'):    ['Wheat','Barley'],
        ('chalky','spring'):    ['Vegetables','Sunflower'],
        ('chalky','autumn'):    ['Sesame','Pulses'],
    }

    # Typical climate ranges by season (avg temp °C, typical rainfall mm)
    season_climate = {
        'monsoon': {'temp_mean': 26, 'temp_sd': 3, 'rain_mean': 900, 'rain_sd': 300},
        'summer' : {'temp_mean': 32, 'temp_sd': 3, 'rain_mean': 150, 'rain_sd': 100},
        'winter' : {'temp_mean': 18, 'temp_sd': 4, 'rain_mean': 200, 'rain_sd': 80},
        'spring' : {'temp_mean': 22, 'temp_sd': 3, 'rain_mean': 150, 'rain_sd': 70},
        'autumn' : {'temp_mean': 25, 'temp_sd': 3, 'rain_mean': 300, 'rain_sd': 120},
    }

    rows = []

    # 1) Build systematic base examples from mapping (one per mapping entry)
    for (soil, season), crops in crop_map.items():
        climate = season_climate.get(season, {'temp_mean': 25, 'temp_sd':3, 'rain_mean':300, 'rain_sd':100})
        for crop in crops:
            # create several base rows with different pH to cover variety
            for ph in [5.5, 6.0, 6.5, 7.0, 7.5]:
                temp = round(np.random.normal(climate['temp_mean'], climate['temp_sd']), 1)
                rainfall = max(0, int(np.random.normal(climate['rain_mean'], climate['rain_sd'])))
                rows.append([soil, season, temp, ph, rainfall, crop])

    # 2) Add region-specific extras and manual entries
    manual_examples = [
        ['loamy','monsoon',27,6.5,1200,'Rice'],
        ['loamy','monsoon',28,6.8,900,'Maize'],
        ['black','summer',30,7.0,500,'Cotton'],
        ['sandy','summer',33,6.0,400,'Millet'],
        ['clayey','winter',18,6.8,700,'Wheat'],
        ['clayey','monsoon',25,6.0,900,'Paddy'],
        ['red','summer',29,5.8,350,'Peanuts'],
        ['loamy','winter',20,6.5,300,'Lentils'],
        ['black','monsoon',26,7.2,1100,'Sugarcane'],
        ['sandy','summer',32,6.5,300,'Groundnut'],
        ['loamy','summer',30,6.8,450,'Sorghum'],
        ['red','monsoon',28,6.0,600,'Pulses'],
        ['black','summer',29,7.1,450,'Soybean'],
        ['peaty','monsoon',26,6.4,1000,'Rice'],
        ['chalky','winter',15,7.2,150,'Barley'],
        ['chalky','spring',20,6.8,200,'Sunflower'],
    ]
    rows.extend(manual_examples)

    # 3) Enlarge dataset: jitter around base rows
    base_rows = list(rows)  # snapshot
    for _ in range(n_jitter):
        base = base_rows[np.random.randint(0, len(base_rows))]
        soil, season, temp_b, ph_b, rain_b, crop = base
        # jitter numerical values
        temp = round(max(-5, min(45, float(temp_b) + np.random.randn()*2.5)), 1)
        ph   = round(max(3.0, min(9.0, float(ph_b) + np.random.randn()*0.25)), 1)
        rainfall = max(0, int(float(rain_b) + np.random.randn()*80))
        # occasionally switch to a related crop to add label noise
        if np.random.rand() < 0.08:
            # pick a random crop from same soil-season mapping if exists
            opts = crop_map.get((soil,season), [crop])
            crop = opts[np.random.randint(0, len(opts))]
        rows.append([soil, season, temp, ph, rainfall, crop])

    df = pd.DataFrame(rows, columns=['soil','season','temp','ph','rainfall','crop'])
    # Shuffle rows
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df


def train_and_save(random_state=RANDOM_STATE):
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = build_rich_synthetic_dataset(n_jitter=2500, random_state=random_state)

    X = df[['soil','season','temp','ph','rainfall']]
    y = df['crop']

    cat_cols = ['soil','season']
    num_cols = ['temp','ph','rainfall']

    preproc = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols)
    ])

    clf = Pipeline([
        ('pre', preproc),
        ('rf', RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.12, random_state=random_state)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(f"Model trained. Test accuracy (synthetic): {score:.3f}")

    joblib.dump(clf, MODEL_PATH)
    print(f"Saved demo model to {MODEL_PATH}")
    return clf, df

if __name__ == "__main__":
    train_and_save()
