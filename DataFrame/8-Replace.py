import pandas as pd
import numpy as np

data = {
    "Name": ["Padma", "Ravi", "Sita", "Kiran"],
    "Age": [25, 28, np.nan, 30],
    "Salary": [35000, 45000, np.nan, 50000],
    "Department": ["HR", "IT", "Finance", "IT"]
}

df = pd.DataFrame(data)

df["Name"] = df["Name"].replace("Sita", "Sai")
print(df)
