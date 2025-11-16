import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

SEED = 42

x_data = pd.read_csv('../data/standardized_student_data.csv')
x_data.drop(columns=['Affects_Academic_Performance', 'Mental_Health_Score', 'Addicted_Score', 'Student_ID'], inplace=True)
y_data = pd.read_csv('../data/encoded_student_data.csv')['Affects_Academic_Performance']

x_train, x_test, y_train, y_test = tts(x_data, y_data, test_size=0.2, random_state=SEED, stratify=y_data)

model = SVC(random_state=SEED, class_weight='balanced', kernel='linear')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
plt.title('SVM Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

importances = pd.Series(model.coef_[0], index=x_data.columns).sort_values(ascending=False)

plt.figure(figsize=(12,8))
sns.barplot(x=importances.head(10), y=importances.head(10).index)
plt.title('Top 10 Feature Importance for SVM')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()