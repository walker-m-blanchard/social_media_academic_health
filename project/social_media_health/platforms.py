import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("data/student_data.csv")

counts = data["Most_Used_Platform"].value_counts()
print(counts) 

threshold = 22

data["Condensed_Platforms"] = data["Most_Used_Platform"].apply(
    lambda x: x if counts[x] >= threshold else "Other"
)

condensed_counts = data["Condensed_Platforms"].value_counts()
print(condensed_counts)

mean_scores = data.groupby("Condensed_Platforms")["Mental_Health_Score"].mean()
print(mean_scores)

order = (
    data.groupby("Condensed_Platforms")["Mental_Health_Score"]
        .mean()
        .sort_values(ascending=False)
        .index
)

# Box Plot: Mental Health vs Platform (Condensed)
plt.figure(figsize=(10,6))
sns.boxplot(
    data=data,
    x="Condensed_Platforms",
    y="Mental_Health_Score",
    order=order,
    palette="colorblind"
)
plt.title("Mental Health Score by Platform")
plt.xlabel("Most Used Platform")
plt.ylabel("Mental Health Score")
plt.tight_layout()
plt.show()

# Bar Chart: Mental Health vs Platform (Condensed)
plt.figure(figsize=(10,6))
sns.barplot(
    data=data,
    x="Mental_Health_Score",
    y="Condensed_Platforms",
    order=order,
    ci=None,
    palette="colorblind"
)
plt.title("Average Mental Health Score by Platform")
plt.xlabel("Average Mental Health Score")
plt.ylabel("Platform")
plt.tight_layout()
plt.show()