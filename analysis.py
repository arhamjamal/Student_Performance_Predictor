import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/students.csv")

# Convert result to numerical value
df["result_numeric"] = df["result"].map({
    "Fail": 0,
    "Pass": 1
})

# 1. Study Hours vs Result
plt.scatter(df["study_hours"], df["result_numeric"])

plt.xlabel("Study Hours")
plt.ylabel("Result (0 = Fail, 1 = Pass)")
plt.title("Study Hours vs Student Result")

plt.show()

plt.scatter(df["attendance"], df["result_numeric"])

plt.xlabel("Attendance (%)")
plt.ylabel("Result (0 = Fail, 1 = Pass)")
plt.title("Attendance vs Student Result")

plt.show()