import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

SEED = 42

# Load encoded data
data = pd.read_csv('data/encoded_student_data.csv')

# Drop columns 
data.drop(columns=['Student_ID', 'Country', 'Age'], inplace=True)

# Separate features (X) and label (y)
X = data.drop(columns=['Relationship_Status'])
y = data['Relationship_Status']

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
