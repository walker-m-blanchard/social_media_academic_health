import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split as tts, KFold, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

SEED = 42

x_data = pd.read_csv('../data/standardized_student_data.csv')
x_data.drop(columns=['Affects_Academic_Performance', 'Mental_Health_Score', 'Addicted_Score', 'Student_ID'], inplace=True)
y_data = pd.read_csv('../data/encoded_student_data.csv')['Affects_Academic_Performance']

x_train, x_test, y_train, y_test = tts(x_data, y_data, test_size=0.2, random_state=SEED, stratify=y_data)

model = SVC(random_state=SEED, class_weight='balanced', kernel='linear', C=5)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))

print('SVM Accuracy: ' + str(accuracy_score(y_test, y_pred)))
print('SVM F1 Score: ' + str(f1_score(y_test, y_pred)))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.title('SVM Confusion Matrix', fontsize=20)
plt.xlabel('Predicts Impacts Academics', fontsize=16)
plt.ylabel('Actually Impacts Academics', fontsize=16)
plt.show()

importances = pd.Series(model.coef_[0], index=x_data.columns).sort_values(ascending=False)

plt.figure(figsize=(12,8))
sns.barplot(x=importances.head(10), y=importances.head(10).index)
plt.title('Top 10 Feature Importance for SVM')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
scores = cross_val_score(model, x_data, y_data, cv=kf, scoring='accuracy')

print(f"Individual 10-fold scores: {scores}")
print(f"Mean cross-validation score: {np.mean(scores):.4f}")
print(f"Standard deviation of scores: {np.std(scores):.4f}")