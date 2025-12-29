import pandas as pd

data = {
    "Name": ["Padma", "Ravi", "Sita", "Kiran"],
    "Age": [25, 28, 24, 30],
    "Salary": [35000, 45000, 32000, 50000],
    "Department": ["HR", "IT", "Finance", "IT"]
}

df = pd.DataFrame(data)
print(df)

df["promoted Salary"]=df["Salary"]*10
print(df)