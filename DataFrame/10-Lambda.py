import pandas as pd
import numpy as np

data = {
    "Name": ["Padma", "Ravi", "Sita", "Kiran"],
    "Age": [25, 28, 24, 30],
    "Salary": [35000, 45000, 32000, 50000],
    "Department": ["HR", "IT", "Finance", "IT"]
}

df = pd.DataFrame(data)
print(df)

print("\nUpdate the salary")
df["Updated_Salary"] = df["Salary"].apply(lambda x: x * 1.10)
print(df)


df["Age_Group"] = df["Age"].apply(lambda x: "Young" if x < 28 else "Senior")
print(df)


df["Bonus"] = df.apply(lambda row: row["Salary"] * 0.05 if row["Department"] == "IT" else row["Salary"] * 0.03, axis=1)
print(df)
