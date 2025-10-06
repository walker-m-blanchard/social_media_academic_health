import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

np.random.seed(42)
song_data = pd.read_csv('data/hf_data.csv')

song_data['popularity'] = pd.to_numeric(song_data['popularity'], errors='coerce')
song_data['explicit'] = pd.to_numeric(song_data['explicit'], errors='coerce')
song_data['Is_Popular'] = (song_data['popularity'] >= 75).astype(int)

unpopular_songs = song_data[song_data['popularity'] <= 25]
unpopular_songs = unpopular_songs[unpopular_songs['popularity'] > 0]

shuffling = np.random.permutation(len(unpopular_songs))
shuffled_songs = unpopular_songs.iloc[shuffling].reset_index(drop=True)
num_rows = len(unpopular_songs) // 7
unpopular_songs = shuffled_songs.head(num_rows)

popular_songs = song_data[song_data['popularity'] >= 75]

song_data = pd.concat([unpopular_songs, popular_songs])

metadata = song_data[['danceability', 'energy', 'key', 'mode', 'speechiness', 'instrumentalness', 'liveness', 'tempo',
                      'loudness', 'acousticness', 'valence', 'duration_ms']]
popular = song_data['Is_Popular']

scaler = StandardScaler()
x_data_std = scaler.fit_transform(metadata)

x_train, x_test, y_train, y_test = tts(x_data_std, popular, test_size=0.2, random_state=42)

pca = PCA(n_components=2)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

loadings = pca.components_
loadings_std = pca.components_.T * np.sqrt(pca.explained_variance_)
loading_matrix = pd.DataFrame(loadings_std, index=metadata.columns)
print(loading_matrix)

cov_matrix = np.cov(x_train, rowvar=False)
eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
index = np.argsort(eig_vals)[::-1]
eig_vals = eig_vals[index]

plt.figure()
plt.bar(range(1, len(eig_vals) + 1), eig_vals)
plt.xlabel("PCA")
plt.ylabel("Eigenvalue")
plt.show()

plt.figure()
plt.scatter(x_train_pca[:, 0], x_train_pca[:, 1], c=y_train)
plt.show()

knn = KNeighborsClassifier(n_neighbors=27)
knn.fit(x_train_pca, y_train)

y_pred = knn.predict(x_test_pca)

print('Accuracy: ' + str(accuracy_score(y_test, y_pred)))
print('F1 Score: ' + str(f1_score(y_test, y_pred)))