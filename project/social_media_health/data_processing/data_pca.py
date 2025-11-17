import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

SEED = 42

# Loads data and selects the possible attributes of interest
student_data = pd.read_csv('../data/standardized_student_data.csv')
student_data.drop(columns=['Student_ID'], inplace=True)

# Drops Affects Academic Performance since it will be used as class
# Drops Mental Health Score since it is a subjective self-assessment
x_data = student_data.drop(columns=['Affects_Academic_Performance', 'Mental_Health_Score', 'Addicted_Score'])
y_data = student_data['Affects_Academic_Performance']

x_train, x_test, y_train, y_test = tts(x_data, y_data, test_size=0.2, random_state=SEED, stratify=y_data)

x_train = x_train.to_numpy()
x_train = x_train.astype('float64')

# Calculates eigenvalues
cov_matrix = np.cov(x_train, rowvar=False)
eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
index = np.argsort(eig_vals)[::-1]
eig_vals = eig_vals[index]

# Prints eigenvalues for PCA evaluation
plt.figure(figsize=[12,8])
plt.bar(range(1, len(eig_vals) + 1), eig_vals, color='green')
plt.title('Objective Attributes Variance', fontsize=20)
plt.xlabel("Component", fontsize=16)
plt.ylabel("Eigenvalue", fontsize=16)
plt.show()

# Trains PCA and transforms test data
pca = PCA(n_components=4)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

# Calculates and prints loading matrix from PCA components
loadings_std = pca.components_.T * np.sqrt(pca.explained_variance_)
loading_matrix = pd.DataFrame(loadings_std, index=x_data.columns)
print(loading_matrix)

loading_matrix.to_csv('../data/loading_matrix.csv', index=False)

pca_transformed_x = pd.DataFrame(pca.transform(x_data))
pca_transformed_x.to_csv('../data/pca_transformed_x.csv', index=False)