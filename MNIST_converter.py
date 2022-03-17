import os

import numpy as np
import pandas as pd
from utils.timers import Timer
from utils.inference_utils import events_to_voxel_grid
import struct
import csv


class MyFixedSizeEventReader:
    """
    Reads events from a '.txt' or '.zip' file, and packages the events into
    non-overlapping event windows, each containing a fixed number of events.
    """

    def __init__(self, path_to_event_file, num_events=1000):
        print('Will use fixed size event windows with {} events'.format(num_events))
        print('Output frame rate: variable')
        self.iterator = pd.read_csv(path_to_event_file, delimiter=',', header=None,
                                    names=['x', 'y', 'pol', 't'],
                                    dtype={'x': np.int16, 'y': np.int16, 'pol': np.int16, 't': np.float64},
                                    engine='c',
                                    skiprows=1, chunksize=num_events, memory_map=True)

    def __iter__(self):
        return self

    def __next__(self):
        with Timer('Reading event window from file'):
            event_window = self.iterator.__next__().values
        return event_window


def extract_events_from_binary(path, destiny):
    event_window_iterator = MyFixedSizeEventReader(path)
    for i, event_window in enumerate(event_window_iterator):
        event_window = event_window[:, [3, 0, 1, 2]]
        data = events_to_voxel_grid(event_window, num_bins=5, width=34, height=34)
        # TODO: define the folders in which the arrays will be stored (create a folder for every sequence??)
        # Wait until we see how to train the network with the MS-COCO dataset to make the n-MNIST be similar
        # print(destiny[:-4])


def iterate_over_folder():
    src_folder = "data/MNIST/Test_csv"
    des_folder = "data/MNIST/Test_npy"
    subfolders = os.listdir(src_folder)
    for subfolder in subfolders[0]:
        contents = os.listdir(src_folder + '/' + subfolder)
        for content in [contents[0]]:
            extract_events_from_binary(src_folder + '/' + subfolder + '/' + content,
                                       des_folder + '/' + subfolder + '/' + content)



iterate_over_folder()
#extract_events_from_binary("data/MNIST/Test_csv/0/00004.csv")
