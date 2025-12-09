import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split as tts, KFold, cross_val_score

SEED = 42

# This script is used to model a K-Nearest Neighbors of
# the objective survey data and the academic impact of social media
x_data = pd.read_csv('../data/pca_transformed_x.csv')
y_data = pd.read_csv('../data/encoded_student_data.csv')['Affects_Academic_Performance']

x_train, x_test, y_train, y_test = tts(x_data, y_data, test_size=0.2, random_state=SEED, stratify=y_data)

# K is chosen to be the closest odd number to the sqrt of the training sample size
knn = KNeighborsClassifier(n_neighbors=27)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)

print('KNN Accuracy: ' + str(accuracy_score(y_test, y_pred)))
print('KNN F1 Score: ' + str(f1_score(y_test, y_pred)))
print("Test Sensitivity: ", recall_score(y_test, y_pred))
print("Test Specificity: ", specificity)

kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
scores = cross_val_score(knn, x_data, y_data, cv=kf, scoring='accuracy')

print(f"Individual 10-fold scores: {scores}")
print(f"Mean cross-validation score: {np.mean(scores):.4f}")
print(f"Standard deviation of scores: {np.std(scores):.4f}")

plt.rcParams['font.size'] = 16
plt.figure(figsize=[12, 8], dpi=300)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.xlabel('Predicts Impacts Academics', fontsize=20)
plt.ylabel('Actually Impacts Academics', fontsize=20)
plt.savefig('../figures/knn_matrix.png', dpi=300)