import pandas as pd

emp = pd.DataFrame({
    "EmpID": [101, 102, 103, 104],
    "Name": ["Padma", "Ravi", "Sita", "Kiran"],
    "DeptID": [1, 2, 3, 2]
})

print(emp)


dept = pd.DataFrame({
    "DeptID": [1, 2, 3],
    "Department": ["HR", "IT", "Finance"]
})

print(dept)


print("\nInner join")
inner_join = pd.merge(emp, dept, on="DeptID", how="inner")
print(inner_join)

print("\nLeft Join")
left_join = pd.merge(emp, dept, on="DeptID", how="left")
print(left_join)

print("\nRight Join")
right_join = pd.merge(emp, dept, on="DeptID", how="right")
print(right_join)

print("\nOuter Join")
outer_join = pd.merge(emp, dept, on="DeptID", how="outer")
print(outer_join)


