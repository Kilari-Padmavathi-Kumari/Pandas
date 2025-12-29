import pandas as pd

data = {
    "Name": ["Padma", "Ravi", "Sita", "Kiran"],
    "Age": [25, 28, 24, 30],
    "salary": [35000, 45000, 32000, 50000],
    "Department": ["HR", "IT", "Finance", "IT"]
}

df = pd.DataFrame(data)
print(df)


df["salary"]=df["salary"]+5000
print(df["salary"])