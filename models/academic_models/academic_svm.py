import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split as tts, KFold, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score

SEED = 42

# This script is used to model a Support Vector Machine of
# the objective survey data and the academic impact of social media
x_data = pd.read_csv('../data/standardized_student_data.csv')
x_data.drop(columns=['Affects_Academic_Performance', 'Mental_Health_Score', 'Addicted_Score', 'Student_ID'],
            inplace=True)
y_data = pd.read_csv('../data/encoded_student_data.csv')['Affects_Academic_Performance']

x_train, x_test, y_train, y_test = tts(x_data, y_data, test_size=0.2, random_state=SEED, stratify=y_data)

model = SVC(random_state=SEED, class_weight='balanced', kernel='linear', C=5)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)

print('SVM Accuracy: ' + str(accuracy_score(y_test, y_pred)))
print('SVM F1 Score: ' + str(f1_score(y_test, y_pred)))
print("Test Sensitivity: ", recall_score(y_test, y_pred))
print("Test Specificity: ", specificity)

plt.rcParams['font.size'] = 16
plt.figure(figsize=[12, 8], dpi=300)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.xlabel('Predicts Impacts Academics', fontsize=20)
plt.ylabel('Actually Impacts Academics', fontsize=20)
plt.savefig('../figures/svm_matrix.png', dpi=300)

# Identifies the 5 most important features for the SVM model and compares them with bar graph
importances = pd.Series(model.coef_[0], index=x_data.columns).sort_values(ascending=False)
importances = importances.head(5)
index = {'Conflicts_Over_Social_Media': 'Conflicts', 'Most_Used_Platform_Snapchat': 'Snapchat',
         'Avg_Daily_Usage_Hours': 'Hours of Use', 'Academic_Level': 'Academic Level',
         'Most_Used_Platform_TikTok': 'TikTok'}
importances.rename(index=index, inplace=True)

fig, ax = plt.subplots(figsize=[12, 8], dpi=300)
sns.barplot(x=importances, y=importances.index)
plt.yticks(fontsize=12)
plt.xlabel('Importance', fontsize=20)
plt.ylabel('', fontsize=20)
plt.savefig('../figures/svm_feat_import.png', dpi=300)

kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
scores = cross_val_score(model, x_data, y_data, cv=kf, scoring='accuracy')

print(f"Individual 10-fold scores: {scores}")
print(f"Mean cross-validation score: {np.mean(scores):.4f}")
print(f"Standard deviation of scores: {np.std(scores):.4f}")