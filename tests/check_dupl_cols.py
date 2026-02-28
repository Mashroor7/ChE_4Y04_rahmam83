import pandas as pd

df = pd.read_pickle('data/processed/train_final.pkl')
print(f"Unique column names: {len(df.columns.unique())}")
print(f"Total columns: {len(df.columns)}")
print(f"\nDuplicates:")
dupes = df.columns[df.columns.duplicated(keep=False)]
print(dupes.tolist())