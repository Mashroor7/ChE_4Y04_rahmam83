import numpy as np

data = np.load('data/processed/windows/train_windows_w5_s1.npz', allow_pickle=True)

print("Keys:", list(data.keys()))
print(f"X shape: {data['X'].shape}")
print(f"y shape: {data['y'].shape}")
print(f"Flattened dim: {data['X'].shape[1]}")
print(f"n_features: {data['X'].shape[1] // 5}")
