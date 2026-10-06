import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
df = pd.read_csv("data/students.csv")

# Convert result
df["result"] = df["result"].map({
    "Fail": 0,
    "Pass": 1
})

# Features
X = df[
    [
        "study_hours",
        "attendance",
        "previous_marks",
        "assignment_score"
    ]
]

# Target
y = df["result"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# Scale
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

# Train KNN
model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_train, y_train)

# NEW STUDENT
new_student = pd.DataFrame([{
    "study_hours": 6,
    "attendance": 80,
    "previous_marks": 70,
    "assignment_score": 75
}])

# Scale new student
new_student_scaled = scaler.transform(new_student)

# Predict
prediction = model.predict(new_student_scaled)

if prediction[0] == 1:
    print("Prediction: PASS")
else:
    print("Prediction: FAIL")