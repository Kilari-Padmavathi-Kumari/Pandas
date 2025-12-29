import pandas as pd
import numpy as np

data = {
    "Name": ["Padma", "Ravi", "Sita", "Kiran"],
    "Age": [25, 28, np.nan, 30],
    "Salary": [35000, 45000, np.nan, 50000],
    "Department": ["HR", "IT", "Finance", "IT"]
}

df = pd.DataFrame(data)

print("Before filling null values:")
print(df)

print("\nNull values count:")
print(df.isnull().sum())

# Mean & Median fill
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Salary"] = df["Salary"].fillna(df["Salary"].median())

print("\nAfter mean/median fill:")
print(df)

# Forward fill (modern way)
df["Age"] = df["Age"].ffill()

print("\nAfter forward fill:")
print(df)
