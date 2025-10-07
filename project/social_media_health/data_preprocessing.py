import pandas as pd
from sklearn.preprocessing import LabelEncoder

student_data = pd.read_csv('data/student_data.csv')

label_encoder = LabelEncoder()
student_data['Gender'] = label_encoder.fit_transform(student_data['Gender'])
student_data['Academic_Level'] = label_encoder.fit_transform(student_data['Academic_Level'])
student_data['Most_Used_Platform'] = label_encoder.fit_transform(student_data['Most_Used_Platform'])
student_data['Relationship_Status'] = label_encoder.fit_transform(student_data['Relationship_Status'])
student_data['Affects_Academic_Performance'] = label_encoder.fit_transform(student_data['Affects_Academic_Performance'])
student_data['Country'] = label_encoder.fit_transform(student_data['Country'])

student_data.to_csv('data/encoded_student_data.csv', index=False)