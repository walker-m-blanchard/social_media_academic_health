import pandas as pd
from sklearn.preprocessing import MinMaxScaler

student_data = pd.read_csv('../data/encoded_student_data.csv')

columns_to_normal = ['Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Mental_Health_Score', 'Conflicts_Over_Social_Media', 'Addicted_Score', 'Academic_Level']
scaler = MinMaxScaler()
student_data[columns_to_normal] = scaler.fit_transform(student_data[columns_to_normal])

student_data.to_csv('../data/normalized_student_data.csv', index=False)