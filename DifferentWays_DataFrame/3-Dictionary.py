import pandas as pd
data = [
    {'Name': 'Padma', 'Age': 25, 'City': 'Hyderabad'},
    {'Name': 'Ravi', 'Age': 30, 'City': 'Bangalore'},
    {'Name': 'Anu', 'Age': 28, 'City': 'Chennai'}
]

df = pd.DataFrame(data)
print(df)
