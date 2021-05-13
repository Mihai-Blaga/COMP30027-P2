import pandas as pd
import numpy as np


def create_csv_output(file_name: str, result: np.ndarray):
    output = pd.DataFrame({"duration_label": result})
    output.index += 1
    output.to_csv(file_name + ".csv", index_label="id")
