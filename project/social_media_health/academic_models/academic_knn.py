import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split as tts

SEED = 42

x_data = pd.read_csv('../data/pca_transformed_x.csv')
y_data = pd.read_csv('../data/encoded_student_data.csv')['Affects_Academic_Performance']

x_train, x_test, y_train, y_test = tts(x_data, y_data, test_size=0.2, random_state=SEED, stratify=y_data)

# Fits and predicts with KNN
# K is chosen to be the sqrt of the training sample size
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

print('KNN Accuracy: ' + str(accuracy_score(y_test, y_pred)))
print('KNN F1 Score: ' + str(f1_score(y_test, y_pred)))