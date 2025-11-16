import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

SEED = 42

student_data = pd.read_csv('data/encoded_student_data.csv')

x_data = student_data[['Addicted_Score']]
y_data = student_data[['Affects_Academic_Performance']]

x_train, x_test, y_train, y_test = tts(x_data, y_data, test_size=0.2, random_state=SEED, stratify=y_data)

model = LogisticRegression(random_state=SEED)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print('Accuracy: ' + str(accuracy_score(y_test, y_pred)))
print('F1 Score: ' + str(f1_score(y_test, y_pred)))

kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
scores = cross_val_score(model, x_data, y_data, cv=kf, scoring='accuracy')

print(f"Individual 10-fold scores: {scores}")
print(f"Mean cross-validation score: {np.mean(scores):.4f}")
print(f"Standard deviation of scores: {np.std(scores):.4f}")

x_plot = np.linspace(x_data.min(), x_data.max()).reshape(-1,1)
final_model = LogisticRegression(random_state=SEED)
final_model.fit(x_data, y_data)
y_prob = final_model.predict_proba(x_plot)[:,1]

plt.figure(figsize=[12,8])
plt.plot(x_plot, y_prob)
plt.title('Logistic Regression')
plt.xlabel('Addicted Score')
plt.ylabel('Affects Academic Performance')
plt.show()