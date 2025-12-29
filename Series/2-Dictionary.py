import pandas as pd
fruits_calories = {
    "Apple": 52,
    "Banana": 89,
    "Orange": 47,
    "Mango": 60
}
df=pd.Series(fruits_calories,name='Protein')
print(df)


# Conditional Selection

print("maximum",df[df>30])

#logical operator : & | ~

print(df[(df>20)&(df<80)])

print(df[~(df>80)])


#modify the series

df["Apple"]=20
print(df)