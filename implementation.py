
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
data=pd.read_csv("blood_sample_dataset.csv")
print(data)
le=LabelEncoder()
data['Disease']=le.fit_transform(data['Disease'])
X = data.drop(columns=["Disease"])
y = data["Disease"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
rf_model = RandomForestClassifier(random_state=42)
dt_model = DecisionTreeClassifier(random_state=42)

# Train models
rf_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)
y_pred_dt = dt_model.predict(X_test)

# Evaluation
rf_report = classification_report(y_test, y_pred_rf)
dt_report = classification_report(y_test, y_pred_dt)

print("Random Forest Classification Report:\n", rf_report)
print("Decision Tree Classification Report:\n", dt_report)

# Confusion Matrix Visualization using a Scatter Plot
fig, ax = plt.subplots(figsize=(8, 5))
true_labels, pred_labels = np.unique(y_test), np.unique(y_pred_rf)
ax.scatter(y_test, y_pred_rf, color='blue', alpha=0.6, label='Random Forest Predictions')
ax.scatter(y_test, y_pred_dt, color='orange', alpha=0.6, label='Decision Tree Predictions')
ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
ax.set_xlabel("True Labels")
ax.set_ylabel("Predicted Labels")
ax.set_title("Scatter Plot of True vs Predicted Labels")
ax.legend()
plt.show()

# Additional Visualization: Bar Chart of Model Accuracy
accuracy_scores = [rf_model.score(X_test, y_test), dt_model.score(X_test, y_test)]
models = ["Random Forest", "Decision Tree"]

plt.figure(figsize=(8, 5))
plt.bar(models, accuracy_scores, color=["blue", "orange"])
plt.xlabel("Model")
plt.ylabel("Accuracy Score")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)
for i, v in enumerate(accuracy_scores):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=12)
plt.show()
