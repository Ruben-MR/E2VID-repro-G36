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

- **base_model.py**: contains the definition of the base class for defining the posterior networks, it contains the init, forward and summary functions to be overriden or called by the models.

#### Model

The model is defined in a hierarchical structure, using as reference the BaseModel from **base_model.py**, the model builds upon the UNet class, which is constructed using the elements in **submodules.py**.

- **submodels.py**: contains the definition of the different building blocks used for building the network.
  - **ConvLayer**: convolutional layer with **relu** activation option and batc normalization, instance normalization, or none.
  - **TransposedConvLayer**: similar to the previous one, but using nn.ConvTranspose2d instead of nn.Conv2d.
  - **UpsampleConvLayer**: similar to the ConvLayer class, but it performs **bilinear** interpolation before running the forward pass of the network.
  - **ConvLSTM**: first applies a convolution operation to an input which corresponds to the concatenation through the channel dimension of the input and the previous hidden state. The output of such operation is chunked to obtain the four different gates fo the LSTM, to which the corresponding sigmoid/tanh activation is applied before computing the new cell and hidden states.
  - **ConvGRU**: in contrast to the previous layer, it performs three independent convolutions over the input + hidden_state tensor, one for each of the gates of the layer, the initialization of the weights is an orthogonal one. Then the GRU computation is performed considering the previous hidden state (or creating one if None).
  - **RecurrentConvLayer**: performs a convolution operation using a **ConvLayer** object and then uses the selected recurrency among the two aforementioned.
  - **DownsampleRecurrentConvLayer**: performs a call to the indicated recurrent layer (ConvLSTM or ConvGRU), performs the forward pass and finishes with a bilinear interpolation with a 0.5 factor for downsampling.
  - **ResidualBlock**: implementation of a residual block: applies a convolution to the original input; followed by batch normalization/instance normalization, if so; followed by a ReLU; another convolution and normalization; then, if provided with a downsampling layer, apply downsampling; finally, perform the residual addition and apply one last time ReLU.

- **unet.py**: definition of the general UNet architectures that will be parametrized as indicated in the paper to yield the E2VID netoworks.
  - **BaseUnet**: takes the parameters specified in the constructor (later called in the E2VID class) and assigns the values to the parameters of the class. Important considerations:
   - The skip connection is a summation if it is specified as such. Otherwise it performs a concatenation operation along the channel dimension (1)
   - The **use_upsample_conv** will determine whether an **UpsampleConvLayer** is used instead of **TransposedConvLayer**. The former will be slower but will prevent checkerboard artifacts.
   - **build_resblocks**: will create as many residual blocks as indicated in the constuctor parameters
   - **build_decoders**: will create the decoder blocks (selected previously in the constructor) and will specify the input size attending to the type of skip connection performed (sum or concatenation)
   - **build_prediction_layer**: will create a convolutional layer with input channels depending on the type of skip connection, kernel size of 1 and no activation.
  - **UNet**: base class of the UNet architecture. Defines the head layer to obtain the indicated number of input channels to the encoders and builds them as simple **ConvLayer** and finally builds the residual, decoder and prediction layer using the functions defined in the base class. The forward pass is performed as usual, storing the output of the head and encoder layers for the skip connections, then applying the residual blocks and then the decoder layers prior to which the skip operation is applied. Finally, the prediction layer is applied as well as the sigmoid activation.
  - **UNetRecurrent**:  performs the same initialization as the normal **UNet** class but defines the encoders as a **RecurrentConvLayer** with the block type defined between **ConvLSTM** or **ConvGRU**. Performs the same forward pass as before but now, the previous state is stored as well as the output of the head and encoder layers.

- **model.py**: contains the definition of the E2VID network based on the classes defined in **unet.py**.
  - **BaseE2VID**: base function of the network, its __init__ function defines the following parameters: 
   - num_bins: needs to be passed, otherwise it will launch an assertion error, it corresponds to the number of bins in the voxel grid event tensor. (??) It will determine the number of input channels of the overall network. 
   - skip connection type: default to sum
   - number of encoder blocks: 4
   - initial number of channels: 32 (after the head layer)
   - number of residual blocks: 2
   - normalization: none
   - use upsample convolutional layers: true
  - **E2VID**: class calling the normal UNet architecture with the provided parameters (or the ones defined in the base class by default) and performs the forward pass, outputting the image in black and white.
  - **E2VIDRecurrent**: similar to the previous one, making a call to the UNetRecurrent class with the provided parameters, using a ConvLSTM or ConvGRU after each encoder block.

#### Options

- Function for parsing possible inference arguments passed to the run-reconstruction file upon execution.

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
