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


![Alt text](docs/source/img/data_flow_vqa_5_attention_gpu_loaders.png?raw=true "Exemplary multi-modal data flow diagram")


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
PTP provides several ready to use, out of the box components, from ones of general usage to very specialized ones:

  * Feed Forward Network (Fully Connected layers with activation functions and dropout, variable number of hidden layers, general usage)
  * Torch Vision Wrapper (wrapping several models from Torch Vision, e.g. VGG-16, ResNet-50, ResNet-152, DenseNet-121, general usage)
  * Convnet Encoder (CNNs with ReLU and MaxPooling, can work with different sizes of images)
  * LeNet-5 (classical baseline)
  * Recurrent Neural Network (different kernels with activation functions and dropout, a single model can work both as encoder or decoder, general usage)
  * Seq2Seq (Sequence to Sequence model, classical baseline)
  * Attention Decoder (RNN-based decoder implementing Bahdanau-style attention, classical baseline)
  * Sencence Embeddings (encodes words using embedding layer, general usage)

Currently PTP offers the following models useful for multi-modal fusion and reasoning:

  * VQA Attention (simple question-driven attention over the image)
  * Element Wise Multiplication (Multi-modal Low-rank Bilinear pooling, MLB)
  * Multimodel Compact Bilinear Pooling (MCB)
  * Miltimodal Factorized Bilinear Pooling
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


## Quick start: MNIST image classification with a simple ConvNet model

Please consider a simple ConvNet model consisting of two parts: 
  * few convolutional layers returning _feature maps_ being, in general, a 4D tensor (first dimension being associated with the batch size),
  * one (or more) dense layers that accepts the (flattened) feature maps and return _predictions_ as logarithm of probability distributions (LogSoftmax as last non-linearity).

### Training the model

Assume that we will use ```NLL Loss``` function, and, besides, want to monitor the ```Accuracy``` statistics.
The resulting pipeline is presented below.
The additional ```Answer Decoder``` component translates the predictions into class names, whereas ```Stream Viewer``` displays content of the indicated data streams for a single sample randomly picked from the batch.
The associated ```mnist_classification_convnet_softmax.yml``` configuration file can be found in ```configs/tutorials``` folder.


![Alt text](docs/source/img/1_tutorials/data_flow_tutorial_mnist_1_training.png?raw=true "Trainining of a simple ConvNet model on MNIST dataset")


We will train the model with _ptp-offline-trainer_, a general _worker_ script that follows the classical training-validation, epoch-based methodology.
This means, that despite the presence of three section (associated with training, validation and test splits of the MNIST dataset) the trainer will consider only the content of ``training`` and ```validation``` sections (plus ```pipeline```, containing the definition of the whole pipeline).
Let's run the training by calling the following from the command line:

```console
ptp-offline-trainer --c configs/tutorials/mnist_classification_convnet_softmax.yml
```

__Note__: Please call ```offline-trainer --h``` to learn more about the run-time arguments. In order to understand the structure of the main configuration file please look at the default configuration file of the trainer located in ```configs/default/workers``` folder.

The trainer will log on the console training and validation statistis, along with additional information logged by the components, e.g. contents of the streams:

```console
[2019-07-05 13:31:44] - INFO - OfflineTrainer >>> episode 006000; epoch 06; loss 0.1968410313; accuracy 0.9219
[2019-07-05 13:31:45] - INFO - OfflineTrainer >>> End of epoch: 6
================================================================================
[2019-07-05 13:31:45] - INFO - OfflineTrainer >>> episode 006019; episodes_aggregated 000860; epoch 06; loss 0.1799264401; loss_min 0.0302138925; loss_max 0.5467863679; loss_std 0.0761705562; accuracy 0.94593; accuracy_std 0.02871 [Full Training]
[2019-07-05 13:31:45] - INFO - OfflineTrainer >>> Validating over the entire validation set (5000 samples in 79 episodes)
[2019-07-05 13:31:45] - INFO - stream_viewer >>> Showing selected streams for sample 20 (index: 55358):
 'labels': One
 'targets': 1
 'predictions': tensor([-1.1452e+01, -1.6804e-03, -1.1357e+01, -1.1923e+01, -6.6160e+00,
        -1.4658e+01, -9.6191e+00, -8.6472e+00, -9.6082e+00, -1.3505e+01])
 'predicted_answers': One
```

Please note that whenever the validation loss goes down, the trainer automatically will save the pipeline to the checkpoint file:

```console
[2019-07-05 13:31:47] - INFO - OfflineTrainer >>> episode 006019; episodes_aggregated 000079; epoch 06; loss 0.1563445479; loss_min 0.0299939774; loss_max 0.5055227876; loss_std 0.0854654983; accuracy 0.95740; accuracy_std 0.02495 [Full Validation]
[2019-07-05 13:31:47] - INFO - mnist_classification_convnet_softmax >>> Exporting pipeline 'mnist_classification_convnet_softmax' parameters to checkpoint:
 /users/tomaszkornuta/experiments/mnist/mnist_classification_convnet_softmax/20190705_132624/checkpoints/mnist_classification_convnet_softmax_best.pt
  + Model 'image_encoder' [ConvNetEncoder] params saved
  + Model 'classifier' [FeedForwardNetwork] params saved
```

After the training finsh the trainer will inform about the termination reason and indicate where the experiment files (model checkpoint, log files, statistics etc.) can be found:

```console
[2019-07-05 13:32:33] - INFO - mnist_classification_convnet_softmax >>> Updated training status in checkpoint:
 /users/tomaszkornuta/experiments/mnist/mnist_classification_convnet_softmax/20190705_132624/checkpoints/mnist_classification_convnet_softmax_best.pt
[2019-07-05 13:32:33] - INFO - OfflineTrainer >>>
================================================================================
[2019-07-05 13:32:33] - INFO - OfflineTrainer >>> Training finished because Converged (Full Validation Loss went below Loss Stop threshold of 0.15)
[2019-07-05 13:32:33] - INFO - OfflineTrainer >>> Experiment finished!
[2019-07-05 13:32:33] - INFO - OfflineTrainer >>> Experiment logged to: /users/tomaszkornuta/experiments/mnist/mnist_classification_convnet_softmax/20190705_132624/
```


### Testing the model

In order to test the model generalization we will use _ptp-processor_, yet another general _worker_ script that performs a single pass over the indicated set.


![Alt text](docs/source/img/1_tutorials/data_flow_tutorial_mnist_2_test.png?raw=true "Test of the pretrained model on test split of the MNIST dataset ")


```console
ptp-processor --load /users/tomaszkornuta/experiments/mnist/mnist_classification_convnet_softmax/20190705_132624/checkpoints/mnist_classification_convnet_softmax_best.pt
```

__Note__: _ptp-processor_ uses the content of _test_ section as default, but it can be changed at run-time. Please call ```ptp-processor --h``` to learn about the available run-time arguments.


```console
[2019-07-05 13:34:41] - INFO - Processor >>> episode 000313; episodes_aggregated 000157; loss 0.1464060694; loss_min 0.0352710858; loss_max 0.3801054060; loss_std 0.0669835582; accuracy 0.95770; accuracy_std 0.02471 [Full Set]
[2019-07-05 13:34:41] - INFO - Processor >>> Experiment logged to: /users/tomaszkornuta/experiments/mnist/mnist_classification_convnet_softmax/20190705_132624/test_20190705_133436/
```

__Note__: Please analyse the ```mnist_classification_convnet_softmax.yml``` configuration file (located in ```configs/tutorials``` directory). Keep in mind that:
  * all components come with default configuration files, located in ```configs/default/components``` folders,
  * all workers come with default configuration files, located in ```configs/default/workers``` folders.


## Maintainers

A project of the Machine Intelligence team, IBM Research, Almaden.

* Tomasz Kornuta (tkornut@us.ibm.com)

[![HitCount](http://hits.dwyl.io/tkornut/tkornut/pytorchpipe.svg)](http://hits.dwyl.io/tkornut/tkornut/pytorchpipe)
