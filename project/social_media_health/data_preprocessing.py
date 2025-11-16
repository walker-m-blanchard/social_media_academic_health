import pandas as pd
from sklearn.preprocessing import LabelEncoder

student_data = pd.read_csv('data/student_data.csv')

label_encoder = LabelEncoder()
student_data['Gender'] = label_encoder.fit_transform(student_data['Gender'])
student_data['Affects_Academic_Performance'] = label_encoder.fit_transform(student_data['Affects_Academic_Performance'])

academic_map = {'High School': 0, 'Undergraduate': 1, 'Graduate': 2}
student_data['Academic_Level'] = student_data['Academic_Level'].map(academic_map)

categorical_cols = ['Most_Used_Platform', 'Relationship_Status']
new_cols = pd.get_dummies(student_data, columns=categorical_cols, drop_first=True)

new_student_data = student_data.assign(**new_cols)
new_student_data.drop(columns=['Country', 'Age', 'Most_Used_Platform', 'Relationship_Status'], inplace=True)

new_student_data.to_csv('data/encoded_student_data.csv', index=False)