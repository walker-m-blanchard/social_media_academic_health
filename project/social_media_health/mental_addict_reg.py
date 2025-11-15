import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from collections import Counter

SEED = 42

student_data = pd.read_csv('data/encoded_student_data.csv')

x_data = student_data[['Addicted_Score']]
y_data = student_data[['Mental_Health_Score']]

x_train, x_test, y_train, y_test = tts(x_data, y_data, test_size=0.2, random_state=SEED)

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print('R2 Score: ' + str(r2_score(y_test, y_pred)))

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
scores = cross_val_score(model, x_data, y_data, cv=kf, scoring='r2')

print(f"Individual 5-fold scores: {scores}")
print(f"Mean cross-validation score: {np.mean(scores):.4f}")
print(f"Standard deviation of scores: {np.std(scores):.4f}")

kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
scores = cross_val_score(model, x_data, y_data, cv=kf, scoring='r2')

print(f"Individual 10-fold scores: {scores}")
print(f"Mean cross-validation score: {np.mean(scores):.4f}")
print(f"Standard deviation of scores: {np.std(scores):.4f}")

final_model = LinearRegression()
final_model.fit(x_data, y_data)
y_pred = final_model.predict(x_data)

x_df = student_data['Addicted_Score']
y_df = student_data['Mental_Health_Score']
points = list(zip(x_df, y_df))
freq = Counter(points)

sizes = []
for i in range(len(x_df)):
    sizes.append(freq[(x_df[i], y_df[i])] * 10)

plt.figure(figsize=[12,8])
plt.scatter(x_data, y_data, s=sizes)
plt.plot(x_data, y_pred)
plt.title('Linear Regression')
plt.xlabel('Addicted Score')
plt.ylabel('Mental Health Score')
plt.show()