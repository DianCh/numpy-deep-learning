# numpy-deep-learning
A deep learning framework written from scratch using NumPy.
## Introduction
This is a vanilla deep learning framework which is implemented from scrach using pure NumPy library. The main purpose isn't, of course, to put together yet another powerful auto-grad library (with cpu-only NumPy, seriously?), but instead to document and summarize the math behind the most commonly seen deep learning building blocks when I recently reviewed them.

All the forward pass, backward pass, initialization, update, etc., are implemented with numpy matrix operations; no magic wrappers behind whatsoever. It's a good reference for practitioners in this field (might just be me) to review the basics, and also for people who just started to see how the black box (or more precisely, boxes) works under the scene.

Simply out of personal preference, Pytorch is chosen to be the "ground truth" reference to verify the implementation. Also, the interfaces have a similar style with Pytorch.

## Installation

Dependency:

- NumPy >= 1.18
- scikit-learn >= 0.20.0
- PyTorch (if you want to run the verification. cpu version will do; no need for gpu support.)

Clone the repository and set up python search path:

```bash
git clone https://github.com/DianCh/numpy-deep-learning.git
export PYTHONPATH=<your-chosen-location>/numpy-deep-learning:$PYTHONPATH
```

## Experiments

The deep learning components are packaged into `ndl`, while scripts of different experiments are grouped under `experiments`. Before running the experiments, please run the following the script to get **MNIST** and **CIFAR10** datasets ready on your local machine:

```bash
python prepare_data.py
```

Then you should have the follow folders for your data inside this repo:

```bash
├── data
│   ├── cifar-10-batches-py
│   └── mnist-784
```

If you prefer to put data elsewhere, you can also achieve that by setting `NDL_DATA_ROOT` environment variable.

### Numerical Verification

### Multi-Layer Perceptron

In this experiment, we are going to train a Multi-Layer Perceptron to classify hand-written digits from the MNIST dataset.



### Convolutional Neural Network

### Speed Benchmark


## Math Step-by-Step
### Linear
### Conv2D
### Initialization
### Pool2D
### ReLU
### CrossEntropyLoss


## License