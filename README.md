# This is the README of the blog for the reproducibility project

## Here we can note down things, links or whichever information we may deem relevant as well as modified files to save each other some time and avoid re-implementing the same thing three times

### Useful links

Project's webpage from ETH [link](https://rpg.ifi.uzh.ch/E2VID.html)

### n-MNIST

I have found this [repository](https://github.com/gorchard/event-Python) from Garrick Orchard in which he claims to have a *VERY* initial version of some files for handling event data. However, I think that the authors themselves have included some files for parsing and reading the data from these datasets, but we wil need to try it out.

Btw, I the link to n-MNIST is [here](https://www.garrickorchard.com/datasets/n-mnist).

Synthesizing very briefly the contents of the dataset, it contains 60000 training and 10000 teest samples event data of the same visual scale as the original MNIST (28x28).

Each sample is a separate binary file consisting of a list of events. Each event occupies 40 bits arranged as described below:

bit 39 - 32: Xaddress (in pixels)
bit 31 - 24: Yaddress (in pixels)
bit 23: Polarity (0 for OFF, 1 for ON)
bit 22 - 0: Timestamp (in microseconds)

### Contents of the original repository 

Data and pretrained subfolders are omitted since they only contain the data and the pretrained models, if existent, there is no "original code" in them.

#### Base



#### Model



#### Scripts

- **embed_reconstructed_images_in_robag.py**, **extract_events_from_rosbag.py** and **image_folder_tp_rosbag.py** are used for conversion from rosbag to .txt format, seems unlikely that we'll end up using these.
- **resample_reconstructions.py**: resamples the reconstructed time-stamped images and selects the ones that lead to a fixed, user-defined framerate.

#### Utils

- **path_utils.py**: contains a simple function that verifies the existence of a folder and creates it otherwise
- **loading_utils.py**: contains a function for loading a pretrained model in **run_reconstruction.py** and a function for selecting the device that will be used for computation.
- **timers.py**: pretty self-explanatory, defines CUDA and normal timers for event reading and processing.
- **util.py**: once again, pretty self-explanatory, very small but powerful and efficient functions.
- **inference_utils.py**: utils for image reconstruction, we may need to take a deeper look at it.
  - make_even_preview: ...
- **event_readers**:  

### Possible Future Works

[de Tournemire et al.](https://arxiv.org/abs/2001.08499) have created a large dataset of event-based camera samples, specialised in the automotive sector. It might be interesting for us to take a look at this dataset, and maybe train our model on it.

