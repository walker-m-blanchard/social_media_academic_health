import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

SEED = 42

# Loads data and selects the possible attributes of interest
student_data = pd.read_csv('data/encoded_student_data.csv')
student_data.drop(columns=['Student_ID'], inplace=True)

# Drops Affects Academic Performance since it will be used as class
# Drops Mental Health Score since it is a subjective self-assessment
x_data = student_data.drop(columns=['Affects_Academic_Performance', 'Mental_Health_Score', 'Addicted_Score'])
y_data = student_data['Affects_Academic_Performance']

x_train, x_test, y_train, y_test = tts(x_data, y_data, test_size=0.2, random_state=SEED, stratify=y_data)

# Fits and predicts with a random forest decision tree
rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)

print('RF Accuracy: ' + str(accuracy_score(y_test, y_pred)))
print('RF F1 Score: ' + str(f1_score(y_test, y_pred)))

# Verifies accuracy of random forest across k-fold, with 5 & 10 folds used
kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
scores = cross_val_score(rf, x_data, y_data, cv=kf, scoring='accuracy')

print(f"Individual 10-fold scores: {scores}")
print(f"Mean cross-validation score: {np.mean(scores):.4f}")
print(f"Standard deviation of scores: {np.std(scores):.4f}")

# Fits and displays final decision tree based on entire dataset
final_model = RandomForestClassifier(n_estimators=100, random_state=SEED)
final_model.fit(x_data, y_data)

final_tree = final_model.estimators_[0]
plot_tree(final_tree, feature_names=x_data.columns,
          class_names=['Does Not Affect Academics', 'Affects Academics'], filled=True)
plt.show()

# Calculates and displays the importance of features in the RF model
importances = pd.Series(final_model.feature_importances_, index=x_data.columns).sort_values(ascending=False)

plt.figure(figsize=(18,12))
sns.barplot(x=importances.head(10), y=importances.head(10).index)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# Calculates and prints a more complete version of accuracy measures
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()