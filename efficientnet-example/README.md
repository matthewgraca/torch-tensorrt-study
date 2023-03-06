# Intro

These are notes for learning Torch-TensorRT. I'm going through the Jupyter Notebooks in 
[this](https://github.com/pytorch/TensorRT/tree/main/notebooks) repo.

The goal for this particular notebook is to understand how Torch-TensorRT can be used 
with 3rd party, pre-trained models; the official walkthrough can be found 
[here](https://github.com/pytorch/TensorRT/blob/main/notebooks/EfficientNet-example.ipynb).

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

Another important file we'll download is the set of labels for imagenet, which contains data that 
maps a number (as a string) to a pair, containing an identification string and the name of the 
object associated with that string.

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

Finally, we'll store the labels in a dictionary, mapping an index to a pair containing an 
identification string and the object associated with that string, formatted like so: 
`"86": ["n01807496", "partridge"]`.

```python
with open("./data/imagenet_class_index.json") as json_file:
    d = json.load(json_file)
```

# Utility Functions

Before we continue, we'll define some utility functions that we'll use to preprocess, predict, 
and conduct our benchmarks. After defining these, we'll test them out on our sample.

## Utility Function Definitions

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

### `predict()`
This function is responsible for making the actual prediction.

```python
# decode the results into ([predicted class, description], probability)
def predict(img_path, model):
    img = Image.open(img_path)
    preprocess = efficientnet_preprocess()
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
        # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
        sm_output = torch.nn.functional.softmax(output[0], dim=0)

    ind = torch.argmax(sm_output)
    return d[str(ind.item())], sm_output[ind] #([predicted class, description], probability)
```

Let's analyze the first block of code. The image to predict is opened, preprocessed, converted into 
a `(224, 224, 3)` tensor. 

However, efficientnet requires its inputs to also have an extra dimension for batch size, so it is 
expecting a tensor of the shape `(N, 224, 224, 3)`, where `N` is the batch size. 
[[6]](https://github.com/lukemelas/EfficientNet-PyTorch/issues/174)
By using `tensor.unsqueeze(0)`, we are adding another dimension to `input_tensor` at position `0`.
[[7]](https://stackoverflow.com/a/65831759)

Next, let's take a look at the second block of code. Recall that we moved the model to the GPU; 
here we move both the data and model to ensure both are on the GPU and neither is on the CPU.

Now, for the third block of code. We mark the model to ensure gradient calculations are not being 
made. We don't care about those since we are doing no training, only inference. We then pass the 
input we want to infer about throught the model. The model will return a tensor of numerical values 
of shape 1000, as there are 1000 classes defined in Imagenet's database. To make sense of these 
numbers, we further modify this tensor by converting this tensor of numbers into a tensor of 
probabilities (or confidences) with the softmax function.

You'll notice we used `output[0]`; this is because strictly speaking, our tensor is not 
one-dimensional. In particular, `model(input_batch) returns a tensor of the shape `(1, 1000)`. 

The fourth and final block of code gets the index of the largest value in the softmax tensor, 
which is the label with the highest probability. This is a tensor with one value, so we use 
`torch.tensor.item()` to convert tensors with one item to a standard Python number. Using 
the dictionary of labels we initialized earlier, we can map this index number to a particular 
object in one of our 1000 classes.

### `benchmark()`

This function is responsible for performing our benchmarks.

```python
def benchmark(model, input_shape=(1024, 1, 224, 224), dtype='fp32', nwarmup=50, nruns=10000):
    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda")
    if dtype=='fp16':
        input_data = input_data.half()

    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns+1):
            start_time = time.time()
            features = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i%10==0:
                print('Iteration %d/%d, avg batch time %.2f ms'%(i, nruns, np.mean(timings)*1000))

    print("Input shape:", input_data.size())
    print("Output features size:", features.size())
    print('Average throughput: %.2f images/second'%(input_shape[0]/np.mean(timings)))
```

Let's begin with the first block. We start by making our input data a tensor of random numbers 
in the shape of a tensor that the model will accept. The point is to test performance, so 
we don't care if these are "real" pictures - this will help us test performance without being 
bogged down downloading and converting pictures to our required format for input. We shuffle 
that data into our GPU, and check the data type we are passing in. We'll discuss more about 
this function and how it relates to TensorRT later; just know that we can run models with 
different precisions.

Next, we have to "warm up" our model. As it turns out, the first few inferences that are made 
are significantly slower than the subsequent inferences that are made because the model defers 
initializations and optimizations until its first few inference requests. 
[[8]](https://medium.com/log-loss/model-warmup-8e9681ef4d41) Again, we mark the model to not 
automatically make gradient calculations since we are just doing inference. We warm up the model 
with a default number of passes, with `nwarmup=50` in this case. Furthermore, 
`torch.cuda.synchronize()` prevens the CPU thread from proceeding until all inference is 
completed. [[9]](https://auro-227.medium.com/timing-your-pytorch-code-fragments-e1a556e81f2) 

After warmup is completed, we start the benchmark for actual inference runs. Mark the tensors 
to ensure no gradients are being calculated, then infer what is `input_data` `nrun` times. 
For example, our default `input_shape=(1024, 1, 224, 224)` with `nruns=10000`. That means by 
default we would be making inferences on 1024 images per run; so 10,240,000 total inferences. 
In reality, our tests will be much less rigorous and use smaller shapes and runs. Moving on, 
ensure to capture the time it takes to make each inference, and synchronize the cuda device 
to ensure the serial portion of the code is not continuing before all inference per run is complete. 
Finally, output data on performance after every 10th iteration.

The last set of print statements just show the shape of the input, the shape of the output, 
and the average number of images that complete inference per second.

## Testing `predict()` on the Sample

Here, we'll show off the utility functions on our sample.

```python
for i in range(4):
    img_path = './data/img%d.JPG'%i
    img = Image.open(img_path)

    pred, prob = predict(img_path, efficientnet_b0_model)
    print('{} - Predicted: {}, Probablility: {}'.format(img_path, pred, prob))

    plt.subplot(2,2,i+1)
    plt.imshow(img);
    plt.axis('off');
    plt.title(pred[1])
```

In this snippet, we give `predict` a path to the image and the model we're using, giving us a 
dictionary with the predicted class and description, and the probability of the image matching 
the best prediction the model made.

# Benchmarks

Here, we'll perform the benchmarks without Torch-TensorRT. Then we'll use Torch-TensorRT in FP32 
precision mode, and in FP16 precision mode, and compare the results of the benchmark.

Each benchmark will have an `input_shape=(128, 3, 224, 224)`, so essentially 128 RGB images being 
inferred per run, with `nruns=100`.

## Without Torch-TensorRT

To benchmark this, we simply pass in the model unmodified into `benchmark()` along with our 
agreed input shape and number of runs.

```python
# Model benchmark without Torch-TensorRT
benchmark(model, input_shape=(128, 3, 224, 224), nruns=100)
```

And would produce output similar to this:
```
Warm up ...
Start timing ...
Iteration 10/100, avg batch time 37.62 ms
Iteration 20/100, avg batch time 37.66 ms
Iteration 30/100, avg batch time 37.65 ms
Iteration 40/100, avg batch time 37.66 ms
Iteration 50/100, avg batch time 37.70 ms
Iteration 60/100, avg batch time 37.70 ms
Iteration 70/100, avg batch time 37.70 ms
Iteration 80/100, avg batch time 37.71 ms
Iteration 90/100, avg batch time 37.72 ms
Iteration 100/100, avg batch time 37.72 ms
Input shape: torch.Size([128, 3, 224, 224])
Output features size: torch.Size([128, 1000])
Average throughput: 3393.46 images/second
```

## With Torch-TensorRT: FP32 (Single Precision)

To accelerate our inference, we need to make various changes to the model that we currently have. 
Since our tensors are already in FP32 format, this is a demonstration of only the layer and tensor 
fusions in the model graph, and the kernel library's best implementations for the target GPU.

What is of note is the simplicity of use - one function converts a model to the necessary format; 
in this case, `torch_tensorrt.compile()`. This returns a compiled PyTorch module that will execute 
via TensorRT when run. It takes in the model to compile, the input shape and data type, what 
precision TensorRT can use when selecting kernels. Another argument is the workspace size for 
TensorRT; I don't know much about this other than the fact that it's an int and `1 << 22 = 4194304`.
[[10]](https://pytorch.org/TensorRT/py_api/torch_tensorrt.html#torch-tensorrt-py)

```python
# The compiled module will have precision as specified by "op_precision".
# Here, it will have FP32 precision.
trt_model_fp32 = torch_tensorrt.compile(model, inputs = [torch_tensorrt.Input((128, 3, 224, 224), dtype=torch.float32)],
    enabled_precisions = torch.float32, # Run with FP32
    workspace_size = 1 << 22
)
```

When benchmarking, the code is the same, except for the model passed through:

```python
# Obtain the average time taken by a batch of input
benchmark(trt_model_fp32, input_shape=(128, 3, 224, 224), nruns=100)
```

The output should look like this:

```
Warm up ...
Start timing ...
Iteration 10/100, avg batch time 27.86 ms
Iteration 20/100, avg batch time 27.71 ms
Iteration 30/100, avg batch time 27.99 ms
Iteration 40/100, avg batch time 27.95 ms
Iteration 50/100, avg batch time 27.89 ms
Iteration 60/100, avg batch time 27.85 ms
Iteration 70/100, avg batch time 28.00 ms
Iteration 80/100, avg batch time 27.97 ms
Iteration 90/100, avg batch time 27.95 ms
Iteration 100/100, avg batch time 27.92 ms
Input shape: torch.Size([128, 3, 224, 224])
Output features size: torch.Size([128, 1000])
Average throughput: 4584.06 images/second
```

## With Torch-TensorRT: FP16 (Half Precision)

Again, we'll compile the model to work with TensorRT - but this time, with the data type to be set 
to half precision:

```python
# The compiled module will have precision as specified by "op_precision".
# Here, it will have FP16 precision.
trt_model_fp16 = torch_tensorrt.compile(model, inputs = [torch_tensorrt.Input((128, 3, 224, 224), dtype=torch.half)],
    enabled_precisions = {torch.half}, # Run with FP16
    workspace_size = 1 << 22
)
```

We'll then run the benchmark:

```python
# Obtain the average time taken by a batch of input
benchmark(trt_model_fp16, input_shape=(128, 3, 224, 224), dtype='fp16', nruns=100)
```

Before analyzing the output, recall that `benchmark()` had a special condition for data types at 
FP16:
```python
    if dtype=='fp16':
        input_data = input_data.half()
```

This function converts the data type of our input tensors from `torch.float32` to `torch.float16`. 
[[11]](https://pytorch.org/docs/stable/generated/torch.Tensor.half.html)

Back the output of our benchmark, we should have:
```
Warm up ...
Start timing ...
Iteration 10/100, avg batch time 12.05 ms
Iteration 20/100, avg batch time 12.56 ms
Iteration 30/100, avg batch time 12.39 ms
Iteration 40/100, avg batch time 12.34 ms
Iteration 50/100, avg batch time 12.33 ms
Iteration 60/100, avg batch time 12.32 ms
Iteration 70/100, avg batch time 12.30 ms
Iteration 80/100, avg batch time 12.28 ms
Iteration 90/100, avg batch time 12.35 ms
Iteration 100/100, avg batch time 12.35 ms
Input shape: torch.Size([128, 3, 224, 224])
Output features size: torch.Size([128, 1000])
Average throughput: 10362.23 images/second
```

# Conclusion

We've taken a look at how we can grab a pretrained model, recompile that module into a module 
executed by TensorRT, and benchmarked the performance improvements that were made as a result. 
Of particular note is the ease in which the conversion was possible - one function to compile 
a module was all that was needed for FP32, and while compilation only needed one line for FP16, 
we needed to add an additional condition to change the input tensors from FP32 to FP16 as well.
