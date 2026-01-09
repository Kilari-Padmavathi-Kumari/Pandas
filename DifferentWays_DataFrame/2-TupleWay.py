import pandas as pd
data = [
    ('Padma', 25, 'Hyderabad'),
    ('Ravi', 30, 'Bangalore'),
    ('Anu', 28, 'Chennai')
]

df = pd.DataFrame(data, columns=['Name', 'Age', 'City'])
print(df)

print(df.loc[0])
print(df.loc[1])



