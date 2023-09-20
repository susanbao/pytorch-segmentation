import numpy as np
import json
import os

def read_one_results(path):
    with open(path, "r") as outfile:
        data = json.load(outfile)
    return data

def write_one_results(json_data, path):
    with open(path, "w") as outfile:
        json.dump(json_data, outfile)
        
def np_read(file):
    with open(file, "rb") as outfile:
        data = np.load(outfile)
    return data
def np_write(data, file):
    with open(file, "wb") as outfile:
        np.save(outfile, data)
        
def check_folder_exist(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)