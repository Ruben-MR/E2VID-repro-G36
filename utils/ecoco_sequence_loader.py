"""

This is a set of sequence loader functions for the event based MS-COCO dataset

"""
import json
import os.path
import os
import glob
import numpy as np
import pandas as pd
import cv2
from config import DATA_DIR


def load_flow(sequence_type, sequence_number, path=DATA_DIR):
    filepath = os.path.join(path,
                            "ecoco_depthmaps_test",
                            sequence_type,
                            "sequence_{:>010d}".format(sequence_number),
                            "flow")

    os.chdir(filepath)

    boundary_timestamps = pd.read_csv("boundary_timestamps.txt", header=None, delim_whitespace=True)
    boundary_timestamps.columns = ("idx", "t1", "t2")

    flow = []

    for file in glob.glob("*.npy"):
        flow.append(np.load(file))

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    return flow, boundary_timestamps


def load_frames(sequence_type, sequence_number, path=DATA_DIR):
    filepath = os.path.join(path,
                            "ecoco_depthmaps_test",
                            sequence_type,
                            "sequence_{:>010d}".format(sequence_number),
                            "frames")

    os.chdir(filepath)

    timestamps = pd.read_csv("timestamps.txt", header=None, delim_whitespace=True)
    timestamps.columns = ("idx", "t")

    with open("params.json") as f:
        params = json.load(f)

    images = []

    for file in glob.glob("*.png"):
        images.append(cv2.imread(file))

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    return images, timestamps, params


def load_voxelgrid(sequence_type, sequence_number, path=DATA_DIR):
    filepath = os.path.join(path,
                            "ecoco_depthmaps_test",
                            sequence_type,
                            "sequence_{:>010d}".format(sequence_number),
                            "VoxelGrid-betweenframes-5")

    os.chdir(filepath)

    boundary_timestamps = pd.read_csv("boundary_timestamps.txt", header=None, delim_whitespace=True)
    boundary_timestamps.columns = ("idx", "t1", "t2")

    timestamps = pd.read_csv("timestamps.txt", header=None, delim_whitespace=True)
    timestamps.columns = ("idx", "t")

    with open("params.json") as f:
        params = json.load(f)

    event_tensors = []

    for file in glob.glob("*.npy"):
        event_tensors.append(np.load(file))

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    return event_tensors, timestamps, boundary_timestamps, params


def load_everyting(sequence_type, sequence_number, path=DATA_DIR):

    flow, f_boundary_timestamps = load_flow(sequence_type, sequence_number, path=path)
    images, i_timestamps, i_params = load_frames(sequence_type, sequence_number, path=path)
    event_tensors, e_timestamps, e_boundary_timestamps, e_params = load_voxelgrid(sequence_type, sequence_number, path=path)

    sequence_dict = {"flow": {"data": flow,
                              "boundary_timestamps": f_boundary_timestamps},
                     "frames": {"data": images,
                                "timestamps": i_timestamps,
                                "params": i_params},
                     "VoxelGrid-betweenframes-5": {"data": event_tensors,
                                                   "timestamps": e_timestamps,
                                                   "boundary_timestamps": e_boundary_timestamps,
                                                   "params": e_params}}

    return sequence_dict


if __name__ == '__main__':

    sequence_type = "train"     # Either 'train' or 'validation'
    sequence_number = 18         # int, [0-949] with 'train', [950-1000] with 'validation'

    # flow, boundary_timestamps = load_flow(sequence_type, sequence_number)
    # print(f"{flow[0].shape}")
    # print(flow[0])
    # print(boundary_timestamps)

    # images, timestamps, params = load_frames(sequence_type, sequence_number)
    # cv2.imshow('', images[0])
    # cv2.waitKey()
    # print(timestamps)
    # print(json.dumps(params, indent=4))

    # event_tensors, timestamps, boundary_timestamps, params = load_voxelgrid(sequence_type, sequence_number)
    # print(event_tensors[0])
    # print(event_tensors[0].shape)
    # print(timestamps)
    # print(boundary_timestamps)
    # print(json.dumps(params, indent=4))

    sequence_dict = load_everyting(sequence_type, sequence_number)

    print(sequence_dict["flow"]["data"])
    print(sequence_dict["flow"]["boundary_timestamps"])
    print(sequence_dict["frames"]["data"])
    print(sequence_dict["frames"]["timestamps"])
    print(sequence_dict["frames"]["params"])
    print(sequence_dict["VoxelGrid-betweenframes-5"]["data"])
    print(sequence_dict["VoxelGrid-betweenframes-5"]["timestamps"])
    print(sequence_dict["VoxelGrid-betweenframes-5"]["boundary_timestamps"])
    print(sequence_dict["VoxelGrid-betweenframes-5"]["params"])
