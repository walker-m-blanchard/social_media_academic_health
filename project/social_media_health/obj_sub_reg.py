import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

SEED = 42

student_data = pd.read_csv('data/standardized_student_data.csv')
student_data.drop(columns=['Student_ID'], inplace=True)

x_data = None
y_labels = ['Affects_Academic_Performance', 'Mental_Health_Score', 'Addicted_Score']

for i in range(2):
    if i == 0:
        x_data = student_data.drop(columns=['Affects_Academic_Performance', 'Mental_Health_Score', 'Addicted_Score'])
        scaler = StandardScaler()
        x_data = scaler.fit_transform(x_data)
        print('Original Data\n--------------------')
    else:
        x_data = pd.read_csv('data/pca_transformed_x.csv')
        print('PCA Data\n--------------------')

    for label in y_labels:
        y_data = student_data[label]
        x_train, x_test, y_train, y_test = tts(x_data, y_data, test_size=0.2, random_state=SEED)

        if label != 'Affects_Academic_Performance':
            model = LinearRegression()
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            print(label, '\n--------------------\nR2 Score: ', str(r2_score(y_test, y_pred)), '\n')

            kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
            scores = cross_val_score(model, x_data, y_data, cv=kf, scoring='r2')

            print(f"Individual 10-fold scores: {scores}")
            print(f"Mean cross-validation score: {np.mean(scores):.4f}")
            print(f"Standard deviation of scores: {np.std(scores):.4f}\n")
        else:
            model = LogisticRegression(random_state=SEED)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            print(label, '\n--------------------')
            print('Accuracy: ' + str(accuracy_score(y_test, y_pred)))
            print('F1 Score: ' + str(f1_score(y_test, y_pred)))

            kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
            scores = cross_val_score(model, x_data, y_data, cv=kf, scoring='accuracy')

            print(f"Individual 10-fold scores: {scores}")
            print(f"Mean cross-validation score: {np.mean(scores):.4f}")
            print(f"Standard deviation of scores: {np.std(scores):.4f}")