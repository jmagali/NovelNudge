import pandas as pd

# Fetch the unprocessed csv file
df = pd.read_csv('unprocessedData.csv')

# Fill null data with empty string
df = df.fillna("")

print(df.head(5))