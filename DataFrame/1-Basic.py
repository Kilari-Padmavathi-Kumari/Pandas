import pandas as pd

data = {
    "Fruit": ["Apple", "Banana", "Orange", "Mango"],
    "Price": [120, 40, 60, 150],
    "Quantity": [10, 25, 15, 8]
}

df = pd.DataFrame(data)
print(df)

#head and tail
print(df.head(2))
print(df.tail(1))

#loc and iloc
print(df.iloc[1:3])  # iloc used for all col printing

print(df.iloc[1:3,:2])  # give two col only

print(df.loc[2:3,['Price','Quantity']])  # particular col printing using loc

print(df[['Price','Fruit']])   # print particular col

#drop


print(df.drop('Price',axis=1))

print(df)


#shape

print(df.shape)

print(df.info())

print(df.describe())