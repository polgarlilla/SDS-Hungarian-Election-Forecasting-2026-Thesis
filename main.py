# Import functions
from data import load_data, aggregate_to_oevk

# Load dataframes
df_oevk, df_national, df_ep, df_polls = load_data()

# Process: Aggregate to OEVK level for the Hungarian parliamentary elections
df_oevk = aggregate_to_oevk(df_oevk)

# Inspect
print(df_oevk.head())
print(df_national.head())
print(df_ep.head())
print(df_polls.head())