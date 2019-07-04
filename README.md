# PyTorchPipe

![Language](https://img.shields.io/badge/language-Python-blue.svg)
[![GitHub license](https://img.shields.io/github/license/IBM/pytorchpipe.svg)](https://github.com/IBM/pytorchpipe/blob/develop/LICENSE)
[![GitHub version](https://badge.fury.io/gh/IBM%2Fpytorchpipe.svg)](https://badge.fury.io/gh/IBM%2Fpytorchpipe)

[![Build Status](https://travis-ci.com/IBM/pytorchpipe.svg?branch=develop)](https://travis-ci.com/IBM/pytorchpipe)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/IBM/pytorchpipe.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/IBM/pytorchpipe/context:python)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/IBM/pytorchpipe.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/IBM/pytorchpipe/alerts/)
[![Coverage Status](https://coveralls.io/repos/github/IBM/pytorchpipe/badge.svg?branch=develop)](https://coveralls.io/github/IBM/pytorchpipe?branch=develop)
[![Maintainability](https://api.codeclimate.com/v1/badges/e8d37123b856ee5bb10b/maintainability)](https://codeclimate.com/github/IBM/pytorchpipe/maintainability)

## Description

PyTorchPipe (PTP) is a component-oriented framework that facilitates development of computational _multi-modal pipelines_ and comparison of diverse neural network-based models.

PTP frames training and testing procedures as _pipelines_ consisting of many components communicating through data streams.
Each such a stream can consist of several components, including one task instance (providing batches of data), any number of trainable components (models) and additional components providing required transformations and computations.

As a result, the training & testing procedures are no longer pinned to a specific task or model, and built-in mechanisms for compatibility checking (handshaking), configuration and global variables management & statistics collection facilitate rapid development of complex pipelines and running diverse experiments.

In its core, to _accelerate the computations_ on their own, PTP relies on PyTorch and extensively uses its mechanisms for distribution of computations on CPUs/GPUs, including multi-process data loaders and multi-GPU data parallelism.
The models are _agnostic_ to those operations and one indicates whether to use them in configuration files (data loaders) or by passing adequate run-time arguments (--gpu).

**Datasets:**
PTP focuses on multi-modal reasoning combining vision and language. Currently it offers the following _Tasks_ from the following task domains:

  * CLEVR, GQA, ImageCLEF VQA-Med 2019 (Visual Question Answering)
  * MNIST, CIFAR-100 (Image Classification)
  * WiLY (Language Identification)
  * WikiText-2 / WikiText-103 (Language Modelling)
  * ANKI (Machine Translation)

Aside of providing batches of samples, the Task class will automatically download the files associated with a given dataset (as long as the dataset is publicly available).
The diversity of those tasks (and associated models) proves the flexibility of the framework, we are working on incorporation of new ones into PTP.

**Pipelines:**
What people typically define as a _model_ in PTP is framed as a _pipeline_, consisting of many inter-connected components, with one or more _Models_ containing trainable elements.
Those components are loosely coupled and care only about the _input streams_ they retrieve and _output streams_ they produce.
The framework offers full flexibility and it is up to the programmer to choose the _granularity_ of his/her components/models/pipelines.
Such a decomposition enables one to easily combine many components and models into pipelines, whereas the framework supports loading of pretrained models, freezing during training, saving them to checkpoints etc.

**Model/Component Zoo:**
PTP provides several ready to use, out of the box components, from ones of general usage to very specialised ones:

  * Feed Forward Network (Fully Connected layers with activation functions and dropout, variable number of hidden layers, general usage)
  * Torch Vision Wrapper (wrapping several models from Torch Vision, e.g. VGG-16, ResNet-50, ResNet-152, DenseNet-121, general usage)
  * Convnet Encoder (CNNs with ReLU and MaxPooling, can work with different sizes of images)
  * LeNet-5 (classical baseline)
  * Recurrent Neural Network (different kernels with activation functions and dropout, a single model can work both as encoder or decoder, general usage)
  * Seq2Seq (Sequence to Sequence model, classical baseline)
  * Attention Decoder (RNN-based decoder implementing Bahdanau-style attention, classical baseline)
  * Sentence Embeddings (encodes words using embedding layer, general usage)

Currently PTP offers the following models useful for multi-modal fusion and reasoning:

  * VQA Attention (simple question-driven attention over the image)
  * Element Wise Multiplication (Multi-modal Low-rank Bilinear pooling, MLB)
  * Multimodel Compact Bilinear Pooling (MCB)
  * Multimodal Factorized Bilinear Pooling
  * Relational Networks

The framework also offers several components useful when working with text:

  * Sentence Tokenizer
  * Sentence Indexer
  * Sentence One Hot Encoder
  * Label Indexer
  * BoW Encoder
  * Word Decoder

and several general-purpose components, from tensor transformations (List to Tensor, Reshape Tensor, Reduce Tensor, Concatenate Tensor), to components calculating losses (NLL Loss) and statistics (Accuracy Statistics, Precision/Recall Statistics, BLEU Statistics etc.) to viewers (Stream Viewer, Stream File Exporter etc.).

**Workers:**
PTP workers are python scripts that are _agnostic_ to the tasks/models/pipelines that they are supposed to work with.
Currently framework offers three workers:

  * ptp-offline-trainer (a trainer relying on classical methodology interlacing training and validation at the end of every epoch, creates separate instances of training and validation tasks and trains the models by feeding the created pipeline with batches of data, relying on the notion of an _epoch_)

  * ptp-online-trainer (a flexible trainer creating separate instances of training and validation tasks and training the models by feeding the created pipeline with batches of data, relying on the notion of an _episode_)

  * ptp-processor (performing one pass over the all samples returned by a given task instance, useful for collecting scores on test set, answers for submissions to competitions etc.)


## Installation

PTP relies on [PyTorch](https://github.com/pytorch/pytorch), so you need to install it first.
Please refer to the official installation [guide](https://github.com/pytorch/pytorch#installation) for details.
It is easily installable via conda_, or you can compile it from source to optimize it for your machine.

PTP is not (yet) available as a [pip](https://pip.pypa.io/en/stable/quickstart/) package, or on [conda](https://anaconda.org/pytorch/pytorch).
However, we provide the `setup.py` script and recommend to use it for installation.
First please clone the project repository:

```console
git clone git@github.com:IBM/pytorchpipe.git
cd pytorchpipe/
```

Next, install the dependencies by running:

```console
  python setup.py develop
```

This command will install all dependencies via pip_, while still enabling you to change the code of the existing components/workers and running them by calling the associated ``ptp-*`` commands.
More in that subject can be found in the following blog post on [dev_mode](https://setuptools.readthedocs.io/en/latest/setuptools.html#development-mode).


## Maintainers

A project of the Machine Intelligence team, IBM Research, Almaden.

* Tomasz Kornuta (tkornut@us.ibm.com)

[![HitCount](http://hits.dwyl.io/tkornut/tkornut/pytorchpipe.svg)](http://hits.dwyl.io/tkornut/tkornut/pytorchpipe)
