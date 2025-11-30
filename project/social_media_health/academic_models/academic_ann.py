import os
import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

SEED = 42

# Set seeds for reproducibility 
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

from tensorflow.keras import Sequential, layers, callbacks
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

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
        epochs = 50,
        batch_size = 5
    )

# Evaluate
y_pred = np.round(model.predict(X_test))

print("Test Accuracy: ", accuracy_score(y_test, y_pred))
print("Test F1 Score: ", f1_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.xlabel('Predicts Impacts Academics', fontsize=16)
plt.ylabel('Actually Impacts Academics', fontsize=16)
plt.title('ANN Confusion Matrix', fontsize=20)
plt.show()

corr = data.corr()['Affects_Academic_Performance'].sort_values(ascending=False)
print(corr)
train_color = '#6B8F4E'
val_color = '#234B03'

plt.plot(history.history['loss'], label='Train Loss', color=train_color)
plt.plot(history.history['val_loss'], label='Val Loss', color=val_color)
plt.legend() 
plt.title('Training vs Validation Loss (50 Epochs)') 
plt.xlabel('Epoch', fontsize = 16)
plt.ylabel('Loss', fontsize = 16)
plt.show()
