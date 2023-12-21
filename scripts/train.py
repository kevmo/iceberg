import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier


data_path = '../data/processed/train.csv'
titanic_data = pd.read_csv(data_path)

X = titanic_data.drop('Survived', axis=1)
y = titanic_data['Survived']

rf_model = RandomForestClassifier()
rf_model.fit(X, y)

models_directory = Path('../models')
models_directory.mkdir(exist_ok=True)
model_filename = f"{models_directory}/RFC_model_0001.joblib"

joblib.dump(rf_model, model_filename)