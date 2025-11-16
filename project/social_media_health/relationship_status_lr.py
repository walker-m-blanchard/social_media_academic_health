import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

SEED = 42

# Load data
# original data
raw = pd.read_csv("data/student_data.csv")

# encoded
data = pd.read_csv('data/encoded_student_data.csv')

le = LabelEncoder()

# Separate features (X) and label (y)
X = data.drop(columns=['Relationship_Status_Complicated',
    'Relationship_Status_In Relationship',
    'Relationship_Status_Single'])
y = le.fit_transform(raw["Relationship_Status"])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

# Build Logistic Regression
lr = LogisticRegression(random_state=SEED, 
                        solver='lbfgs',
                        multi_class='multinomial')
lr.fit(X_train, y_train)

# Evaluate
y_pred = lr.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='g', cmap='Purples')
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
