# Intro

These are notes for learning Torch-TensorRT. I'm going through the Jupyter Notebooks in 
[this](https://github.com/pytorch/TensorRT/tree/main/notebooks) repo.

The goal for this particular notebook is to understand how Torch-TensorRT can be used 
with 3rd party, pre-trained models.

Torch-TensorRT does two major optimizations:
- Fuses layers and tensors in the model graph, using a large kernel library to pick 
implementations that work best on the target GPU
- Offers strong support for reducing the operating precision execution to better leverage 
tensor cores, as well as reducing memory and computation footprints

Essentially, Torch-TensorRT is a compiler that uses TensorRT to optimize TorchScript code; 
compiling TorchScript modules that can internally run with TensorRT optimizations. 
This allows us to stay in the PyTorch ecosystem.

For this notebook, we will demonstrate the steps required to compile a TorchScript module 
with Torch-TensorRT on a pre-trained EfficientNet network, and benchmark speedup results.

As a note, the model is provided by PyTorch's model repository called `timm` - a source for high 
quality computer vision models. EfficientNet is a model pretrained on ImageNet database.

# Preparing the Model

First, we gather our imports and install `timm` with pip:

```bash
pip install timm==0.4.12
```

```python
import torch
import torch_tensorrt
import timm
import time
import numpy as np
import torch.backends.cudnn as cudnn
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import json
```

Then, we initialize our model:

```python
efficientnet_b0_model = timm.create_model('efficientnet_b0',pretrained=True)
model = efficientnet_b0_model.eval().to("cuda")
```

Note that models have two modes: training and evaluation mode. In particular, marking a 
model as `model.train()` informs a few layers to behave differently. On the other hand, 
`model.eval()` freezes these changes. Clearly, when you are training a model, it should 
be marked with `train()`, and when just performing inferenence with no additional training, 
marked as `eval()`. [[1]](https://stackoverflow.com/a/51433411)

As for `to("cuda")`, this method ensures that the model and data are both on a GPU with CUDA; 
since data and model being on different devices would cause a runtime error 
(e.g. data on CPU and model on GPU). [[2]](https://stackoverflow.com/a/63076091)

# Preparing the Data

For our pretrained model, it expects that the data it will be making inferences on is 
formatted in a particular manner.

Specifically, input images are expected to be normalized the same way, with mini-batches of 
3-channel RGB images:
- In the shape of `(3 x H x W)`, where `H` and `W` are expected to be at least `224`
- Images loaded in to a range of `[0,1]`, then normalized using:
  - `mean = [0.485, 0.456, 0.406]`
  - `std = [0.229, 0.224, 0.225]`

## Sample Execution

First we'll download a few images that we'll be transforming.

```bash
$ mkdir -p ./data
$ wget  -O ./data/img0.JPG "https://d17fnq9dkz9hgj.cloudfront.net/breed-uploads/2018/08/siberian-husky-detail.jpg?bust=1535566590&width=630"
$ wget  -O ./data/img1.JPG "https://www.hakaimagazine.com/wp-content/uploads/header-gulf-birds.jpg"
$ wget  -O ./data/img2.JPG "https://www.artis.nl/media/filer_public_thumbnails/filer_public/00/f1/00f1b6db-fbed-4fef-9ab0-84e944ff11f8/chimpansee_amber_r_1920x1080.jpg__1920x1080_q85_subject_location-923%2C365_subsampling-2.jpg"
$ wget  -O ./data/img3.JPG "https://www.familyhandyman.com/wp-content/uploads/2018/09/How-to-Avoid-Snakes-Slithering-Up-Your-Toilet-shutterstock_780480850.jpg"
```

Another important file we'll download is the set of labels for imagenet:

```bash
$ wget  -O ./data/imagenet_class_index.json "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
```

Next, our imports to perform the transformations and show them off:

```python
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
```

Then, we'll perform the transforms on our four downloaded images:

```python
fig, axes = plt.subplots(nrows=2, ncols=2)

for i in range(4):
    img_path = './data/img%d.JPG'%i
    img = Image.open(img_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(img)
    plt.subplot(2,2,i+1)
    plt.imshow(img)
    plt.axis('off')
```

Here we are getting the image, transforming the image to our desired form, showing them off, 
and converting them to tensors for our model. More on the Compose class can be found here. 
[[3]](https://pytorch.org/vision/main/generated/torchvision.transforms.Compose.html)

Finally, we'll load the labels.

```python
with open("./data/imagenet_class_index.json") as json_file:
    d = json.load(json_file)
```

## Utility Functions
Before we continue, we'll define some utility functions that we'll use to preprocess, predict, 
and conduct our benchmarks. After defining these, we'll test them out on our sample.

First, we go into benchmark mode. This is useful when our input sizes don't change, and what we 
get out of it is `cudnn` will look for the optimal set of algorithms for our particular 
configuration, leading to faster runtime -- but only if our input sizes aren't changing. 
[[4]](https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/3)

```python
cudnn.benchmark = True
```

Now, let's define our functions.

### `efficientnet_preprocess()`
This function preprocesses the input images. The model comes with a default data config; this 
contains a URL for the model pretrained weights, number of classes to classify, input image 
size, etc. We pull from this config to preprocess efficientnet. 
[[5]](https://timm.fast.ai/tutorial_feature_extractor#Default-config)

```python
def efficientnet_preprocess():
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    return transform
```
