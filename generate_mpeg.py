import pandas as pd
import numpy as np
import os

desc_list = ["CLD", "CSD", "DCD", "EHD", "HTD", "SCD"]

for desc in desc_list:
    try:
        os.mkdir(f"mpeg_{desc}")
    except:
        pass
    with open(f"mpeg/{desc}.txt") as file:
        for line in file:
            arr = line.split()
            img_name = arr.pop(0)
            file_name = img_name.split(".")[0]
            data = np.asarray(arr, dtype="float32")
            np.savetxt(f"mpeg_{desc}/{file_name}.txt", data)
