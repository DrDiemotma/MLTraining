import numpy as np
import pandas as pd

from MLDemo.DataPulling import puller

def shuffle_and_split_data(data: pd.DataFrame, test_ratio: float):
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1 (exclusive)")
    shuffled_indices: np.ndarray = np.random.permutation(len(data))
    test_set_size: int = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

