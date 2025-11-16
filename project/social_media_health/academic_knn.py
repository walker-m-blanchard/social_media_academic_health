import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts

SEED = 42

# Loads data and selects the possible attributes of interest
student_data = pd.read_csv('data/encoded_student_data.csv')
student_data.drop(columns=['Student_ID', 'Country', 'Age'], inplace=True)

# Drops Affects Academic Performance since it will be used as class
# Drops Mental Health Score since it is a subjective self-assessment
x_data = student_data.drop(columns=['Affects_Academic_Performance', 'Mental_Health_Score', 'Addicted_Score'])
y_data = student_data['Affects_Academic_Performance']

scaler = StandardScaler()
x_data_std = scaler.fit_transform(x_data)

x_train, x_test, y_train, y_test = tts(x_data_std, y_data, test_size=0.2, random_state=SEED, stratify=y_data)

# Fits and predicts with KNN
# K is chosen to be the sqrt of the training sample size
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

print('KNN Accuracy: ' + str(accuracy_score(y_test, y_pred)))
print('KNN F1 Score: ' + str(f1_score(y_test, y_pred)))