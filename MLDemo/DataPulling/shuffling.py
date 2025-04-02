import math
import os
import numpy as np
import pandas as pd
import random



def shuffle_and_split_data(data: pd.DataFrame, test_ratio: float):
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1 (exclusive)")
    shuffled_indices: np.ndarray = np.random.permutation(len(data))
    test_set_size: int = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

def to_randomized_files(data: pd.DataFrame, output_directory: str, file_pattern="train_{:04d}.csv",
                        number_of_files: int = 10, seed: int | None = None) -> list[str]:
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory, exist_ok=True)


    shuffled_list: list[int] = list(range(len(data)))
    random.seed(seed)
    random.shuffle(shuffled_list)

    items_per_batch: int = int(math.ceil(len(data) / number_of_files))
    output_list: list[str] = []

    for i in range(number_of_files):
        i_start = i * items_per_batch
        i_end = (i + 1) * items_per_batch
        if i_end >= len(data):
            i_end = len(data) - 1
        batch_indices: list[int] = shuffled_list[i_start:i_end]
        batch: pd.DataFrame = data.loc[batch_indices]

        file_path = os.path.join(output_directory, file_pattern.format(i))
        batch.to_csv(file_path)
        output_list.append(file_path)

    return output_list





if __name__ == "__main__":
    from MLDemo.DataPulling import puller, HOUSING

    housing_data = puller.open_tgz(HOUSING)[0][1]
    to_randomized_files(housing_data, ".tmp/foo")

