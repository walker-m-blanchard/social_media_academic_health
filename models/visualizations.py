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

# How many students are addicted to social media pie chart
# Group addiction scores
data["Addiction_Level"] = pd.cut(
    data["Addicted_Score"],
    bins=[0, 3, 6, 10],
    labels=["Low", "Moderate", "High"]
)

counts = data["Addiction_Level"].value_counts().reindex(["Low", "Moderate", "High"])

labels = ["Low", "Moderate", "High"]

colors = ["#C9DAB8", "#A8C686", "#6B8F4E"]

plt.figure(figsize=(6,6))
plt.pie(
    counts,
    autopct="%.1f%%",
    colors=colors,
)
plt.legend(labels, title="Level of Addiction", loc="upper right")
plt.title("Percentage of Students Addicted to Social Media")
plt.show()


# Avergae Daily Usage Hours
plt.figure(figsize=(8,5))
sns.histplot(
    data=data,
    x="Avg_Daily_Usage_Hours",
    bins=10,
    kde=True,
    color="#6B8F4E"
)
plt.axvline(data['Avg_Daily_Usage_Hours'].mean(), 
            color="#234B03", linestyle='--', 
            label=f'Mean: {data["Avg_Daily_Usage_Hours"].mean():.1f} hours')
plt.title("Average Daily Social Media Usage Hours")
plt.xlabel("Average Daily Usage Hours")
plt.ylabel("Number of Students")
plt.legend()
plt.tight_layout()
plt.show()

# Usage Hours by Platform
counts = data["Most_Used_Platform"].value_counts()
print(counts) 

threshold = 22

data["Condensed_Platforms"] = data["Most_Used_Platform"].apply(
    lambda x: x if counts[x] >= threshold else "Other"
)

order = (
    data.groupby("Condensed_Platforms")["Avg_Daily_Usage_Hours"]
        .mean()
        .sort_values(ascending=False)
        .index
)

plt.figure(figsize=(10,5))
sns.barplot(
    data=data,
    x="Condensed_Platforms",
    y="Avg_Daily_Usage_Hours",
    order=order,
    palette="Greens",
    ci=None
)
plt.title("Average Usage Hours per Platform")
plt.xlabel("Most Used Platform")
plt.ylabel("Average Daily Usage Hours")
plt.show()

