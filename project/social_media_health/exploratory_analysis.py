import random
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
import matplotlib.pyplot as plt
from collections import Counter

SEED = 42

def mental_health(x_data, y_data):
    model = LinearRegression()
    model.fit(x_data, y_data)
    y_pred = model.predict(x_data)
    r2 = round(r2_score(y_data, y_pred), 3)

    x_df = x_data['Addicted_Score']
    y_df = y_data['Mental_Health_Score']
    points = list(zip(x_df, y_df))
    freq = Counter(points)

    sizes = []
    for i in range(len(x_df)):
        sizes.append(freq[(x_df[i], y_df[i])] * 10)

    plt.figure(figsize=[12,8])
    plt.scatter(x_data, y_data, s=sizes, color='#234B03')
    plt.plot(x_data, y_pred, color='green')
    plt.title('Social Media Addiction and Mental Health', fontsize=20)
    plt.xlabel('Social Media Addiction Score', fontsize=16)
    plt.ylabel('Mental Health Score', fontsize=16)
    plt.annotate('R2 Score: ' + str(r2), (7,9))
    plt.savefig('../figures/addiction_and_mental_regression.png', dpi=300)

def academic(x_data, y_data):
    model = LogisticRegression()
    model.fit(x_data, y_data)
    y_pred = model.predict(x_data)

    accuracy = round(accuracy_score(y_data, y_pred), 3)
    f1 = round(f1_score(y_data, y_pred), 3)

    x_plot = np.linspace(x_data.min(), x_data.max()).reshape(-1, 1)
    y_prob = model.predict_proba(x_plot)[:, 1]

    plt.figure(figsize=[12, 8])
    plt.plot(x_plot, y_prob, color='green')
    plt.title('Social Media Addiction and Academics', fontsize=20)
    plt.xlabel('Social Media Addiction Score', fontsize=16)
    plt.ylabel('Probability Academics Affected', fontsize=16)
    plt.annotate('Accuracy: ' + str(accuracy), (2,0.9))
    plt.annotate('F1: ' + str(f1), (2,0.8))
    plt.savefig('../figures/addiction_and_academic_regression.png', dpi=300)

def main(seed):
    random.seed(seed)

    student_data = pd.read_csv('../data/encoded_student_data.csv')

    addict_data = student_data[['Addicted_Score']]
    mh_data = student_data[['Mental_Health_Score']]
    academic_data = student_data[['Affects_Academic_Performance']]

    mental_health(addict_data, mh_data)
    academic(addict_data, academic_data)

if __name__ == '__main__':
    main(SEED)