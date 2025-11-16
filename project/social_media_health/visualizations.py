import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("data/student_data.csv")

# Most Popular Social Media Platforms Among Students
plt.figure(figsize=(8,5))
sns.countplot(
    data=data,
    y="Most_Used_Platform",
    order=data["Most_Used_Platform"].value_counts().index,
    palette="colorblind"
)
plt.title("Most Popular Social Media Platforms Among Students")
plt.xlabel("Number of Students")
plt.ylabel("Platform")
plt.tight_layout()
plt.show()