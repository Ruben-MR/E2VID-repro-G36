"""
A simple Configuration file. We can use it to specify certain paths in the project structure, or relative paths which lie
outside the project (this might be useful for extracting data from outside the project structure if it is stored on a separate location for example)
"""

import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))      # the root directory of the project
DATA_DIR = '/media/richard/TOSHIBA EXT/TU_Delft/CS4240_DeepLearning/project_dataset'
#DATA_DIR = os.path.join(ROOT_DIR, "data")
SAVED_DIR = os.path.join(ROOT_DIR, "saved_models")

# if __name__ == '__main__':
    # print(f"{ROOT_DIR=}")
    # print(f"{DATA_DIR=}")
    # print(f"{SAVED_DIR=}")
