import os
import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, recall_score
import tensorflow as tf
from tensorflow.keras import Sequential, layers, callbacks
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

SEED = 42

# This script is used to model an Artificial Neural Network of
# the objective survey data and the academic impact of social media

# Set seeds for reproducibility 
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Load encoded data
data = pd.read_csv('../data/normalized_student_data.csv')

# Drop columns that do no affect predictability 
data.drop(columns=['Student_ID'], inplace=True)

# Separate features (X) and label (y)
X = data.drop(columns=['Affects_Academic_Performance', 'Mental_Health_Score', 'Addicted_Score'])
y = data['Affects_Academic_Performance']

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

# Build
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    layers.Dropout(0.3),
    Dense(32, activation='relu'),
    layers.Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Compile
model.compile(
        optimizer=Adam(),
        loss='binary_crossentropy',
        metrics=['accuracy'])

# Train
history = model.fit(
        X_train, y_train,
        validation_split = 0.2,
        epochs = 20,
        batch_size = 5
    )

# Evaluate
y_pred = np.round(model.predict(X_test))
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)

print("Test Accuracy: ", accuracy_score(y_test, y_pred))
print("Test F1 Score: ", f1_score(y_test, y_pred))
print("Test Sensitivity: ", recall_score(y_test, y_pred))
print("Test Specificity: ", specificity)
print("Classification Report:\n", classification_report(y_test, y_pred))

plt.rcParams['font.size'] = 16
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=[12, 8], dpi=300)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.xlabel('Predicts Impacts Academics', fontsize=20)
plt.ylabel('Actually Impacts Academics', fontsize=20)
plt.savefig('../figures/ann_matrix.png', dpi=300)

corr = data.corr()['Affects_Academic_Performance'].sort_values(ascending=False)
print(corr)
train_color = '#6B8F4E'
val_color = '#234B03'

# Creates train-validation loss regression
plt.figure(figsize=[12, 8], dpi=300)
plt.plot(history.history['loss'], label='Train Loss', color=train_color)
plt.plot(history.history['val_loss'], label='Val Loss', color=val_color)
plt.legend()
plt.xlabel('Epoch', fontsize = 20)
plt.ylabel('Loss', fontsize = 20)
plt.savefig('../figures/train_val_loss.png', dpi=300)