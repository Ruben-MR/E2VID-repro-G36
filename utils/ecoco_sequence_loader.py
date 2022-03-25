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
import torch

#from config import DATA_DIR

DATA_DIR = '/home/richard/Q3/Deep_Learning/ruben-mr.github.io/data'


def load_sequence_flow(sequence_type, sequence_number, path=DATA_DIR):
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


def load_sequence_frames(sequence_type, sequence_number, path=DATA_DIR):
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


def load_sequence_event_tensor(sequence_type, sequence_number, path=DATA_DIR):
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


def load_everything_sequence(sequence_type, sequence_number, path=DATA_DIR):

    flow, f_boundary_timestamps = load_sequence_flow(sequence_type, sequence_number, path=path)
    images, i_timestamps, i_params = load_sequence_frames(sequence_type, sequence_number, path=path)
    event_tensors, e_timestamps, e_boundary_timestamps, e_params = load_sequence_event_tensor(sequence_type, sequence_number, path=path)

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


def full_event_tensor(sequence_indices, streamlength, path=DATA_DIR):
    """
    This function creates an event tensor, ready to be used for training.

    :param sequence_indices: list of ints, corresponding to the sequence indices to create the batch, length of the list is the Batch Size
    :param streamlength: int, specifies the number of sequential event tensors to make
    :param path: str, the path wherein the root folder of the COCO dataset is located
    :return:
        - a numpy array, shape (streamlength, batch size, 5, 180, 240), contains the event tensors making up the full tensor
        - a numpy array, shape (streamlength, batch size, 2), contains the retrieved boundary timestamps from the 'boundary_timestamps.txt' file
        - a numpy array, shape (streamlength, batch size), contains the retrieved timestamps from the 'timestamps.txt' file
    """

    event_tensor = np.empty([len(sequence_indices), streamlength, 5, 180, 240])
    boundary_timestamps_tensor = np.empty([len(sequence_indices), streamlength, 2])
    timestamps_tensor = np.empty([len(sequence_indices), streamlength])

    for batch_idx, seq_idx in enumerate(sequence_indices):
        filepath = os.path.join(path,
                                "ecoco_depthmaps_test",
                                "train",
                                "sequence_{:>010d}".format(seq_idx),
                                "VoxelGrid-betweenframes-5")
        os.chdir(filepath)

        boundary_timestamps = pd.read_csv("boundary_timestamps.txt", header=None, delim_whitespace=True)
        boundary_timestamps_tensor[batch_idx] = boundary_timestamps.values[:streamlength,1:]

        timestamps = pd.read_csv("timestamps.txt", header=None, delim_whitespace=True)
        timestamps_tensor[batch_idx] = timestamps.values[:streamlength, -1]

        for stream_index in range(streamlength):
            event_tensor[batch_idx, stream_index] = np.load("event_tensor_{:>010d}.npy".format(stream_index))

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    return event_tensor.transpose([1, 0, 2, 3, 4]), boundary_timestamps_tensor.transpose([1, 0, 2]), timestamps_tensor.transpose([1, 0])


def full_image_tensor(sequence_indices, streamlength, path=DATA_DIR):
    """
    This function creates a tensor containing images retrieved from the corresponding sequence indices, and streamlengths.

    :param sequence_indices: list of ints, corresponding to the sequence indices to create the batch, length of the list is the Batch Size
    :param streamlength: int, specifies the number of sequential event tensors to make
    :param path: str, the path wherein the root folder of the COCO dataset is located
    :return:
        - a numpy array, shape (streamlength, batch size, 180, 240), contains the images
        - a numpy array, shape (streamlength, batch size), contains the retrieved timestamps from the 'timestamps.txt' file
    """

    image_tensor = np.empty([len(sequence_indices), streamlength, 180, 240], dtype='uint8')
    timestamps_tensor = np.empty([len(sequence_indices), streamlength])

    for batch_idx, seq_idx in enumerate(sequence_indices):
        filepath = os.path.join(path,
                                "ecoco_depthmaps_test",
                                "train",
                                "sequence_{:>010d}".format(seq_idx),
                                "frames")
        os.chdir(filepath)

        timestamps = pd.read_csv("timestamps.txt", header=None, delim_whitespace=True)
        timestamps_tensor[batch_idx] = timestamps.values[:streamlength, -1]

        for stream_index in range(streamlength):
            image_tensor[batch_idx, stream_index] = cv2.imread("frame_{:>010d}.png".format(stream_index), cv2.IMREAD_GRAYSCALE)

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    return image_tensor.transpose([1, 0, 2, 3]), timestamps_tensor.transpose([1, 0])


def full_flow_tensor(sequence_indices, streamlength, path=DATA_DIR):
    """
    This function creates a flow tensor, ready to be used for training.

    :param sequence_indices: list of ints, corresponding to the sequence indices to create the batch, length of the list is the Batch Size
    :param streamlength: int, specifies the number of sequential event tensors to make
    :param path: str, the path wherein the root folder of the COCO dataset is located
    :return:
        - a numpy array, shape (streamlength, batch size, 2, 180, 240), contains the flow tensors making up the full tensor
        - a numpy array, shape (streamlength, batch size, 2), contains the retrieved boundary timestamps from the 'boundary_timestamps.txt' file
    """

    flow_tensor = np.empty([len(sequence_indices), streamlength, 2, 180, 240])
    boundary_timestamps_tensor = np.empty([len(sequence_indices), streamlength, 2])

    for batch_idx, seq_idx in enumerate(sequence_indices):
        filepath = os.path.join(path,
                                "ecoco_depthmaps_test",
                                "train",
                                "sequence_{:>010d}".format(seq_idx),
                                "flow")
        os.chdir(filepath)

        boundary_timestamps = pd.read_csv("boundary_timestamps.txt", header=None, delim_whitespace=True)
        boundary_timestamps_tensor[batch_idx] = boundary_timestamps.values[:streamlength, 1:]

        for stream_index in range(1, streamlength):
            flow_tensor[batch_idx, stream_index] = np.load("disp01_{:>010d}.npy".format(stream_index))

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    return flow_tensor.transpose([1, 0, 2, 3, 4]), boundary_timestamps_tensor.transpose([1, 0, 2])


if __name__ == '__main__':

    sequence_type = "train"     # Either 'train' or 'validation'
    sequence_number = 8         # int, [0-949] with 'train', [950-1000] with 'validation'

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

    #sequence_dict = load_everything_sequence(sequence_type, sequence_number)
    events = full_event_tensor([1, 2], 3, DATA_DIR)
    print(events)

    # print(len(sequence_dict["flow"]["data"]))
    # print(sequence_dict["flow"]["boundary_timestamps"])
    # print(sequence_dict["frames"]["data"])
    # print(sequence_dict["frames"]["timestamps"])
    # print(sequence_dict["frames"]["params"])
    # print(sequence_dict["VoxelGrid-betweenframes-5"]["data"])
    # print(sequence_dict["VoxelGrid-betweenframes-5"]["timestamps"])
    # print(sequence_dict["VoxelGrid-betweenframes-5"]["boundary_timestamps"])
    # print(sequence_dict["VoxelGrid-betweenframes-5"]["params"])

    # sequence_indices = [42, 64]
    # streamlength = 8

    # e_tensor, bdry_tmstmps, tmstmps = full_event_tensor(sequence_indices, streamlength)
    # print(e_tensor.shape)
    # print(bdry_tmstmps.shape)
    # print(tmstmps.shape)

    # img_tensor, tmstmps = full_image_tensor(sequence_indices, streamlength)
    # print(tmstmps.shape)
    # print(img_tensor.shape)
    # cv2.imshow('', img_tensor[0, 0])
    # cv2.waitKey()

    # f_tensor, bdry_tmstmps = full_flow_tensor(sequence_indices, streamlength)
    # print(f_tensor.shape)
    # print(bdry_tmstmps.shape)



