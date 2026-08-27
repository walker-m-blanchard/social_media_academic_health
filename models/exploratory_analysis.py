import random
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
import matplotlib.pyplot as plt
from collections import Counter

SEED = 42

# Plots BSMAS addict score vs mental health score
def mental_health(x_data, y_data):
    model = LinearRegression()
    model.fit(x_data, y_data)
    y_pred = model.predict(x_data)
    r2 = round(r2_score(y_data, y_pred), 3)

    # Counts frequency of each score pair to plot frequency along with linear regression
    x_df = x_data['Addicted_Score']
    y_df = y_data['Mental_Health_Score']
    points = list(zip(x_df, y_df))
    freq = Counter(points)

    sizes = []
    for i in range(len(x_df)):
        sizes.append(freq[(x_df[i], y_df[i])] * 10)

    plt.rcParams['font.size'] = 16
    plt.figure(figsize=[12,8])
    plt.scatter(x_data, y_data, s=sizes, color='#234B03')
    plt.plot(x_data, y_pred, color='green')
    plt.xlabel('Social Media Addiction Score', fontsize=20)
    plt.ylabel('Mental Health Score', fontsize=20)
    plt.annotate('R2 Score: ' + str(r2), (7,9))
    plt.savefig('figures/addiction_and_mental_regression.png', dpi=300)

# Plots BSMAS addict score vs academic impact of social media use
def academic(x_data, y_data):
    model = LogisticRegression()
    model.fit(x_data, y_data)
    y_pred = model.predict(x_data)

    accuracy = round(accuracy_score(y_data, y_pred), 3)
    f1 = round(f1_score(y_data, y_pred), 3)

    x_plot = np.linspace(x_data.min(), x_data.max()).reshape(-1, 1)
    y_prob = model.predict_proba(x_plot)[:, 1]

    plt.rcParams['font.size'] = 16
    plt.figure(figsize=[12, 8])
    plt.plot(x_plot, y_prob, color='green')
    plt.xlabel('Social Media Addiction Score', fontsize=20)
    plt.ylabel('Probability Academics Affected', fontsize=20)
    plt.annotate('Accuracy: ' + str(accuracy), (2,0.9))
    plt.annotate('F1: ' + str(f1), (2,0.8))
    plt.savefig('figures/addiction_and_academic_regression.png', dpi=300)

# This script is used to create figures for exploratory analysis of data
# Compares the BSMAS social media addiction score with both
# the mental health score and the academic impact of social media use
# Work being done to convert script into module
def main(seed):
    random.seed(seed)

    student_data = pd.read_csv('data/encoded_student_data.csv')

    addict_data = student_data[['Addicted_Score']]
    mh_data = student_data[['Mental_Health_Score']]
    academic_data = student_data[['Affects_Academic_Performance']]

    mental_health(addict_data, mh_data)
    academic(addict_data, academic_data)

if __name__ == '__main__':
    main(SEED)