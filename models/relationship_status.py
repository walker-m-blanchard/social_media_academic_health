import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("data/student_data.csv")

order = ["Complicated", "In Relationship", "Single"]

# Relationship Status vs Mental Health Score 
plt.figure(figsize=(8,5))
sns.boxplot(
    data=data,
    x="Relationship_Status",
    y="Mental_Health_Score",
    order=order,
    palette="colorblind"
)
plt.title("Mental Health Score by Relationship Status")
plt.xlabel("Relationship Status")
plt.ylabel("Mental Health Score")
plt.tight_layout()
plt.show()


# Relationship Status vs Average Daily Usage Hours 
plt.figure(figsize=(8,5))
sns.boxplot(
    data=data,
    x="Relationship_Status",
    y="Avg_Daily_Usage_Hours",
    order=order,
    palette="colorblind"
)
plt.title("Social Media Usage by Relationship Status")
plt.xlabel("Relationship Status")
plt.ylabel("Average Daily Usage Hours")
plt.tight_layout()
plt.show()


# Usage and Mental Health by Relationship Status
plt.figure(figsize=(8,6))
for label, group in data.groupby("Relationship_Status"):
    corr = group["Avg_Daily_Usage_Hours"].corr(group["Mental_Health_Score"])
    print(f"{label}: correlation = {corr:.3f}")

    sns.regplot(
        data=group,
        x="Avg_Daily_Usage_Hours",
        y="Mental_Health_Score",
        scatter=False,
        label=label,
        ci=None
    )
plt.grid(alpha=0.2)
plt.legend()
plt.title("Usage and Mental Health by Relationship Status")
plt.xlabel("Average Daily Usage Hour")
plt.ylabel("Mental Health Score")
plt.show()

# Usage and Mental Health by Relationship Status
plt.figure(figsize=(8,6))
for label, group in data.groupby("Relationship_Status"):
    corr = group["Conflicts_Over_Social_Media"].corr(group["Mental_Health_Score"])
    print(f"{label}: correlation = {corr:.3f}")

    sns.regplot(
        data=group,
        x="Conflicts_Over_Social_Media",
        y="Mental_Health_Score",
        scatter=False,
        label=label,
        ci=None
    )
plt.grid(alpha=0.2)
plt.legend()
plt.title("Conflicts Over Social Media and Mental Health by Relationship Status")
plt.xlabel("Conflicts Over Social Media")
plt.ylabel("Mental Health Score")
plt.show()