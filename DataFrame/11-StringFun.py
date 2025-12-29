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

print("\nName convert into Upper Case")
df["Name"] = df["Name"].str.upper()
print(df)


df["Department"] = df["Department"].str.replace("IT", "Information Tech")
print(df)

print("\n Find length")
df["Name_Length"] = df["Name"].str.len()
print(df)

df["Starts_with_P"] = df["Name"].str.startswith("P")
print(df)

df["Initial"] = df["Name"].apply(lambda x: x[0])
print(df)

