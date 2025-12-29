import pandas as pd

data = {
    'Name': ['Padma', 'Ravi', 'Anu'],
    'Age': [25, 30, 28],
    'City': ['Hyderabad', 'Bangalore', 'Chennai']
}

df = pd.DataFrame(data)
print(df)



'''df = pd.read_csv('data.csv')       # CSV file
df = pd.read_excel('data.xlsx')    # Excel file'''
