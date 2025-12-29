import pandas as pd

# Create data
data = {
    "Name": ["Padma", "Mani", "Ravi"],
    "Age": [22, 25, 23],
    "City": ["Hyderabad", "Bangalore", "Chennai"]
}

# Create DataFrame
df = pd.DataFrame(data)

# Display DataFrame
print(df)
print(df.shape)
rows,columns=df.shape
print("columns",columns)
print("rows",rows)
print("head",df.head)
print("head",df[2:5])
print("age",df.Age)
print("columns",df.columns)
print(type(df['Age']))
print(df[['Age','City']])
print("maximum age",df['Age'].max())
print(df.describe())
print(df[df.Age>=20])
print(df[df.Age==df['Age'].max()])
print(df.index)
print(df.set_index('Age'))