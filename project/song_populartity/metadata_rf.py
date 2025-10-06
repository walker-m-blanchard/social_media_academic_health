import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

song_data = pd.read_csv('data/selected_data.csv')

# Separates data into metadata (x_data) and binary popularity (y_data)
metadata = song_data.drop(columns=['Is_Popular', 'track_genre'])
popular = song_data['Is_Popular']

x_train, x_test, y_train, y_test = tts(metadata, popular, test_size=0.2, random_state=42, stratify=popular)

# Fits and predicts Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)

print('RF Accuracy: ' + str(accuracy_score(y_test, y_pred)))
print('RF F1 Score: ' + str(f1_score(y_test, y_pred)))