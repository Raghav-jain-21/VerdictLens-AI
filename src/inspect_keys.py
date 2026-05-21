# Save as `inspect_keys.py` - RUN THIS FIRST (5 seconds)
import pandas as pd

act_key = pd.read_csv('justice_data/csv/keys/keys/act_key.csv')
print("act_key COLUMNS:", act_key.columns.tolist())
print("\nFirst 5 ROWS:")
print(act_key.head())
print("\nFamily acts preview:")
act_key['act'] = act_key['act'].astype(str)
family = act_key[act_key['act'].str.contains('125|MARRIAGE|HINDU|GUARDIAN|MAINTENANCE', case=False, na=False)]
print(family.head(10))
