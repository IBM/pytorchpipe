# PyTorchPipe

![Language](https://img.shields.io/badge/language-Python-blue.svg)
[![GitHub license](https://img.shields.io/github/license/IBM/pytorchpipe.svg)](https://github.com/IBM/pytorchpipe/blob/develop/LICENSE)
[![GitHub version](https://badge.fury.io/gh/IBM%2Fpytorchpipe.svg)](https://badge.fury.io/gh/IBM%2Fpytorchpipe)

[![Build Status](https://travis-ci.com/IBM/pytorchpipe.svg?branch=develop)](https://travis-ci.com/IBM/pytorchpipe)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/IBM/pytorchpipe.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/IBM/pytorchpipe/context:python)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/IBM/pytorchpipe.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/IBM/pytorchpipe/alerts/)
[![Coverage Status](https://coveralls.io/repos/github/IBM/pytorchpipe/badge.svg?branch=develop)](https://coveralls.io/github/IBM/pytorchpipe?branch=develop)

## Description

PyTorchPipe (PTP) is component-oriented framework that fosters the development of computational _multi-modal pipelines_ and comparison of diverse neural network-based models. 

PTP frames training and testing procedures as _pipelines_ consisting of many components communicating through data streams.
Each such a stream can consist of several components, including one problem instance (providing batches of data), any number of trainable components (models) and additional components providing required transformations and computations.

As a result, the training & testing procedures are no longer pinned to a specific problem or model, and built-in mechanisms for compatibility checking (handshaking), configuration management & statistics collection facilitate running diverse experiments.

In its core, to _accelerate the computations_ on their own, PTP relies on PyTorch and extensively uses its mechanisms for distribution of computations on CPUs/GPUs, including multi-threaded data loaders and multi-GPU data parallelism.
More importantly, the models are agnostic to those and one indicates whether to use them in configuration files (data loaders) or by passing run-time arguments (--gpu).

**Datasets:** PTP focuses on multi-modal perpeption combining vision and language. Currently it offers the following _Problems_ from both domains:

  * ImageCLEF VQA-Med 2019 (Visual Question Answering)
  * MNIST (Image Classification)
  * WiLY (Language Identification)
  * WikiText-2 / WikiText-103 (Language Modelling)
  * ANKI (Machine Translation)

Aside of providing batches of samples, the Problem class will automatically download the files associated with a given dataset (as long as the dataset is publicly available).

**Model Zoo:**


**Workers:**


## Installation

PTP relies on [PyTorch](https://github.com/pytorch/pytorch), so you need to install it first.
Refer to the official installation [guide](https://github.com/pytorch/pytorch#installation) for its installation.
It is easily installable via conda_, or you can compile it from source to optimize it for your machine.

PTP is not (yet) available as a [pip](https://pip.pypa.io/en/stable/quickstart/) package, or on [conda](https://anaconda.org/pytorch/pytorch).
However, we provide the `setup.py` script and recommend to use it for installation.
First please clone the project repository::

```console
git clone git@github.com:IBM/pytorchpipe.git
cd pytorchpipe/
```

Then, install the dependencies by running::

```console
  python setup.py develop
```

This command will install all dependencies via pip_, while still enabling you to change the code of the existing components/workers and running them by calling the associated ``ptp-*`` commands.
More in that subject can be found in the following blog post on [dev_mode](https://setuptools.readthedocs.io/en/latest/setuptools.html#development-mode).


## Maintainers

A project of the Machine Intelligence team, IBM Research, Almaden.

* Tomasz Kornuta (tkornut@us.ibm.com)

[![HitCount](http://hits.dwyl.io/tkornut/tkornut/pytorchpipe.svg)](http://hits.dwyl.io/tkornut/tkornut/pytorchpipe)
