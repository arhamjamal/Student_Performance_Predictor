import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. Load dataset
df = pd.read_csv("data/students.csv")


# 2. Convert Pass/Fail into numbers
df["result"] = df["result"].map({
    "Fail": 0,
    "Pass": 1
})


# 3. Features
features = [
    "study_hours",
    "attendance",
    "previous_marks",
    "assignment_score"
]

X = df[features]
y = df["result"]


# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 5. Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 6. Create KNN model
model = KNeighborsClassifier(n_neighbors=3)


# 7. Train model
model.fit(X_train_scaled, y_train)


# 8. Test model
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Fail", "Pass"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# 9. Save model
joblib.dump(model, "model/knn_model.pkl")

# 10. Save scaler
joblib.dump(scaler, "model/scaler.pkl")

print("\nModel saved successfully!")
print("Scaler saved successfully!")