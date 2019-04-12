# PyTorchPipe

![Language](https://img.shields.io/badge/language-Python-blue.svg)
[![GitHub license](https://img.shields.io/github/license/IBM/pytorchpipe.svg)](https://github.com/IBM/pytorchpipe/blob/develop/LICENSE)
[![GitHub version](https://badge.fury.io/gh/IBM%2Fpytorchpipe.svg)](https://badge.fury.io/gh/IBM%2Fpytorchpipe)

[![Build Status](https://travis-ci.com/IBM/pytorchpipe.svg?branch=develop)](https://travis-ci.com/IBM/pytorchpipe)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/IBM/pytorchpipe.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/IBM/pytorchpipe/context:python)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/IBM/pytorchpipe.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/IBM/pytorchpipe/alerts/)
[![Coverage Status](https://coveralls.io/repos/github/IBM/pytorchpipe/badge.svg?branch=develop)](https://coveralls.io/github/IBM/pytorchpipe?branch=develop)
[![Coverage Status](https://coveralls.io/repos/github/IBM/pytorchpipe/badge.svg?branch=master)](https://coveralls.io/github/IBM/pytorchpipe?branch=master)

## Description

PyTorchPipe (PTP) aims at _accelerating reproducible Machine Learning Research_ by fostering the development of computational _pipelines_ and comparison of diverse neural network-based models. 

In its core, to _accelerate the computations_ on their own, PTP relies on PyTorch and extensively uses its mechanisms for distribution of computations on CPUs/GPUs.

PTP frames training and testing procedures as _pipelines_ consisting of many components communicating through data streams.
Each such a stream can consist of several components, including one problem instance (providing batches of data), (zero-or-more) trainable models and (any number of) additional components providing required transformations and computations.

As a result, the training & testing mechanisms are no longer pinned to a specific model or problem, and built-in mechanisms for compatibility checking (handshaking), configuration management & statistics collection facilitate running diverse experiments.

## Description

PTP relies on PyTorch_, so you need to install it first. Refer to the official installation guide_ of PyTorch for its installation.
It is easily installable via conda_, or you can compile it from source to optimize it for your machine.

PTP is not (yet) available as a pip_ package, or on conda_.
However, we provide the `setup.py` script and recommend to use it for installation.
First please clone the project repository::

  git clone git@github.com:IBM/pytorchpipe.git
  cd pytorchpipe/

Then, install the dependencies by running::

  python setup.py install

This command will install all dependencies via pip_.
If you plan to develop and introduce changes, please call the following command instead::

  python setup.py develop

This will enable you to change the code of the existing components/workers and still be able to run them by calling the associated ``ptp-*`` commands.
More in that subject can be found in the following blog post on dev_mode_.

.. _guide: https://github.com/pytorch/pytorch#installation
.. _PyTorch: https://github.com/pytorch/pytorch
.. _conda: https://anaconda.org/pytorch/pytorch
.. _pip: https://pip.pypa.io/en/stable/quickstart/
.. _dev_mode: https://setuptools.readthedocs.io/en/latest/setuptools.html#development-mode

## Mainainers

A project of the Machine Intelligence team, IBM Research, Almaden.

* Tomasz Kornuta (tkornut@us.ibm.com)

[![HitCount](http://hits.dwyl.io/tkornut/tkornut/pytorchpipe.svg)](http://hits.dwyl.io/tkornut/tkornut/pytorchpipe)
