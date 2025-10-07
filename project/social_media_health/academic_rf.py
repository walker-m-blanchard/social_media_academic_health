import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Loads data and selects the possible attributes of interest
student_data = pd.read_csv('data/encoded_student_data.csv')
student_data.drop(columns=['Student_ID', 'Country', 'Age'], inplace=True)

# Drops Affects Academic Performance since it will be used as class
# Drops Mental Health Score since it is a subjective self-assessment
x_data = student_data.drop(columns=['Affects_Academic_Performance', 'Mental_Health_Score'])
y_data = student_data['Affects_Academic_Performance']

x_train, x_test, y_train, y_test = tts(x_data, y_data, test_size=0.2, random_state=42)

# Fits and predicts with a random forest decision tree
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)

print('RF Accuracy: ' + str(accuracy_score(y_test, y_pred)))
print('RF F1 Score: ' + str(f1_score(y_test, y_pred)))

# Displays decision tree produced by random forest
tree = rf.estimators_[0]
plot_tree(tree, feature_names=x_data.columns,
          class_names=['Does Not Affect Academics', 'Affects Academics'], filled=True)
plt.show()

# Verifies accuracy of random forest across k-fold, with 5 & 10 folds used
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(rf, x_data, y_data, cv=kf, scoring='f1')

print(f"Individual 5-fold scores: {scores}")
print(f"Mean cross-validation score: {np.mean(scores):.4f}")
print(f"Standard deviation of scores: {np.std(scores):.4f}")

kf = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(rf, x_data, y_data, cv=kf, scoring='f1')

print(f"Individual 10-fold scores: {scores}")
print(f"Mean cross-validation score: {np.mean(scores):.4f}")
print(f"Standard deviation of scores: {np.std(scores):.4f}")