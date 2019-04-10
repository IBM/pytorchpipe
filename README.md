# PyTorchPipe

![Language](https://img.shields.io/badge/language-Python-blue.svg)
[![GitHub license](https://img.shields.io/github/license/ibm/pytorchpipe.svg)](https://github.com/ibm/pytorchpipe/blob/develop/LICENSE)
[![GitHub version](https://badge.fury.io/gh/ibm%2Fpytorchpipe.svg)](https://badge.fury.io/gh/ibm%2Fpytorchpipe)


[![Build Status](https://travis-ci.com/ibm/pytorchpipe.svg?branch=develop)](https://travis-ci.com/ibm/pytorchpipe)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/ibm/pytorchpipe.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/ibm/pytorchpipe/context:python)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/ibm/pytorchpipe.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/ibm/pytorchpipe/alerts/)
[![Coverage Status](https://coveralls.io/repos/github/ibm/pytorchpipe/badge.svg?branch=develop)](https://coveralls.io/github/ibm/pytorchpipe?branch=develop)


## Description

PyTorchPipe (PTP) aims at _accelerating reproducible Machine Learning Research_ by fostering the development of computational _pipelines_ and comparison of diverse neural network-based models. 

In its core, to _accelerate the computations_ on their own, PTP relies on PyTorch and extensively uses its mechanisms for distribution of computations on CPUs/GPUs.

PTP frames training and testing procedures as _pipelines_ consisting of many components communicating through data streams.
Each such a stream can consist of several components, including one problem instance (providing batches of data), (zero-or-more) trainable models and (any number of) additional components providing required transformations and computations.

As a result, the training & testing mechanisms are no longer pinned to a specific model or problem, and built-in mechanisms for compatibility checking (handshaking), configuration management & statistics collection facilitate running diverse experiments.

## Mainainers

A project of the Machine Intelligence team, IBM Research, Almaden.

* Tomasz Kornuta (tkornut@us.ibm.com)

[![HitCount](http://hits.dwyl.io/tkornut/tkornut/pytorchpipe.svg)](http://hits.dwyl.io/tkornut/tkornut/pytorchpipe)
