import pandas as pd

df = pd.read_pickle(r'data\processed\train_final.pkl')

print(f"{'#':<4} {'Column Name':<40} {'Type'}")
print("-" * 60)
for i, col in enumerate(df.columns, 1):
    col_type = "META" if col in ['Run_ID', 'Target', 'Time'] else "FEATURE"
    print(f"{i:<4} {col:<40} {col_type}")

print(f"\nTotal columns: {len(df.columns)}")
print(f"Unique column names: {len(df.columns.unique())}")