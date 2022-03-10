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

### Possible Future Works

[de Tournemire et al.](https://arxiv.org/abs/2001.08499) have created a large dataset of event-based camera samples, specialised in the automotive sector. It might be interesting for us to take a look at this dataset, and maybe train our model on it.

