import pandas as pd
from sklearn.preprocessing import StandardScaler

student_data = pd.read_csv('../data/encoded_student_data.csv')

columns_to_standard = ['Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Mental_Health_Score', 'Conflicts_Over_Social_Media', 'Addicted_Score', 'Academic_Level']
scaler = StandardScaler()
student_data[columns_to_standard] = scaler.fit_transform(student_data[columns_to_standard])

student_data.to_csv('../data/standardized_student_data.csv', index=False)