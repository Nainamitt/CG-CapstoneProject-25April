import pandas as pd

old_data = pd.read_csv("data/sales_data.csv")
new_data = pd.read_csv("data/new_data.csv")

old_mean = old_data["Sales"].mean()
new_mean = new_data["Sales"].mean()

difference = abs(old_mean - new_mean)

print("Old Mean:", old_mean)
print("New Mean:", new_mean)

if difference > 30:
    print("⚠️ Data Drift Detected")
else:
    print("✅ No Drift")


import pandas as pd

new_data = pd.read_csv("data/new_data.csv")

print(new_data)