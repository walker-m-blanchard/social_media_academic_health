import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

song_data = pd.read_csv('data/selected_data.csv')

# Separates data into metadata (x_data) and binary popularity (y_data)
metadata = song_data.drop(columns=['Is_Popular', 'track_genre'])
popular = song_data['Is_Popular']

# Standardizes x_data
scaler = StandardScaler()
x_data_std = scaler.fit_transform(metadata)

x_train, x_test, y_train, y_test = tts(x_data_std, popular, test_size=0.2, random_state=42)

# Calculates eigenvalues
cov_matrix = np.cov(x_train, rowvar=False)
eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
index = np.argsort(eig_vals)[::-1]
eig_vals = eig_vals[index]

# Prints eigenvalues for PCA evaluation
plt.bar(range(1, len(eig_vals) + 1), eig_vals)
plt.xlabel("PCA")
plt.ylabel("Eigenvalue")
plt.show()

# Trains PCA and transforms test data
pca = PCA(n_components=2)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

# Calculates and prints loading matrix from PCA components
loadings_std = pca.components_.T * np.sqrt(pca.explained_variance_)
loading_matrix = pd.DataFrame(loadings_std, index=metadata.columns)
print(loading_matrix)

# Prints scatter plot of the data on the two new PCA axis
plt.scatter(x_train_pca[:, 0], x_train_pca[:, 1], c=y_train)
plt.show()

# Fits and predicts KNN
# K is chosen to be the sqrt of the training sample size
knn = KNeighborsClassifier(n_neighbors=27)
knn.fit(x_train_pca, y_train)
y_pred = knn.predict(x_test_pca)

print('Accuracy: ' + str(accuracy_score(y_test, y_pred)))
print('F1 Score: ' + str(f1_score(y_test, y_pred)))