import pandas as pd
df=pd.Series([1,23,2,32,23,23])
print(df)
#print(df.index)
print(df.dtype)

print(df[0:4])   
                 #iloc ->location based indexing
print(df.name)
df.name="calories"
print(df)

print(df.loc[3])
print(df.iloc[[1,4,5]])


