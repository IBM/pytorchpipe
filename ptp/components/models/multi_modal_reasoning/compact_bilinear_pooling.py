#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) IBM Corporation 2018
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__author__ = "Tomasz Kornuta"


import torch
import numpy as np

from ptp.components.models.model import Model
from ptp.data_types.data_definition import DataDefinition


class CompactBilinearPooling(Model):
    """
    Element of one of classical baselines for Visual Question Answering.

    The model inputs (question and image encodings) are combined with Compact Bilinear Pooling mechanism.

    Fukui, A., Park, D. H., Yang, D., Rohrbach, A., Darrell, T., & Rohrbach, M. (2016). Multimodal compact bilinear pooling for visual question answering and visual grounding. arXiv preprint arXiv:1606.01847.

    Gao, Y., Beijbom, O., Zhang, N., & Darrell, T. (2016). Compact bilinear pooling. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 317-326).

    Inspired by implementation from:
    https://github.com/DeepInsight-PCALab/CompactBilinearPooling-Pytorch/blob/master/CompactBilinearPooling.py
    """ 
    def __init__(self, name, config):
        """
        Initializes the model, creates the required layers.

        :param name: Name of the model (taken from the configuration file).

        :param config: Parameters read from configuration file.
        :type config: ``ptp.configuration.ConfigInterface``

        """
        super(CompactBilinearPooling, self).__init__(name, CompactBilinearPooling, config)

        # Get key mappings.
        self.key_image_encodings = self.stream_keys["image_encodings"]
        self.key_question_encodings = self.stream_keys["question_encodings"]
        self.key_outputs = self.stream_keys["outputs"]

        # Retrieve input/output sizes from globals.
        self.image_encoding_size = self.globals["image_encoding_size"]
        self.question_encoding_size = self.globals["question_encoding_size"]
        self.output_size = self.globals["output_size"]

        # Initialize sketch projection matrices.
        image_sketch_projection_matrix = self.generate_count_sketch_projection_matrix(self.image_encoding_size, self.output_size)
        question_sketch_projection_matrix = self.generate_count_sketch_projection_matrix(self.question_encoding_size, self.output_size)

        # Make them parameters of the model, so can be stored/loaded and trained (optionally).
        trainable_projections = self.config["trainable_projections"]
        self.image_sketch_projection_matrix = torch.nn.Parameter(image_sketch_projection_matrix, requires_grad=trainable_projections)
        self.question_sketch_projection_matrix = torch.nn.Parameter(question_sketch_projection_matrix, requires_grad=trainable_projections)


    def generate_count_sketch_projection_matrix(self, input_size, output_size):
        """ 
        Initializes Count Sketch projection matrix for given input (size).
        Its role will be to project vector v∈Rn to y∈Rd.
        We initialize two vectors s∈{−1,1}n and h∈{1,...,d}n:
            * s contains either 1 or −1 for each index
            * h maps each index i in the input v to an index j in the output y.
        Both s and h are initialized randomly from a uniform distribution and remain constant.
        """
        # Generate s: 1 or -1
        s = 2 * np.random.randint(2, size=input_size) - 1
        s = torch.from_numpy(s)
        #print("s=",s.shape)

        # Generate h (indices)
        h = np.random.randint(output_size, size=input_size)
        #print("h=",h.shape)
        indices = np.concatenate((np.arange(input_size)[..., np.newaxis],h[..., np.newaxis]), axis=1)
        indices = torch.from_numpy(indices)
        #print("indices=",indices.shape)

        # Generate sparse matrix.
        sparse_sketch_matrix = torch.sparse.FloatTensor(indices.t(), s, torch.Size([input_size, output_size]))
        #print("\n sparse_sketch_matrix=",sparse_sketch_matrix.shape)
        # Return dense matrix.
        dense_ssm = sparse_sketch_matrix.to_dense().type(self.app_state.FloatTensor)
        #print("\n dense_ssm=",dense_ssm)

        return dense_ssm

        

    def input_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {
            self.key_image_encodings: DataDefinition([-1, self.image_encoding_size], [torch.Tensor], "Batch of encoded images [BATCH_SIZE x IMAGE_ENCODING_SIZE]"),
            self.key_question_encodings: DataDefinition([-1, self.question_encoding_size], [torch.Tensor], "Batch of encoded questions [BATCH_SIZE x QUESTION_ENCODING_SIZE]"),
            }


    def output_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {
            self.key_outputs: DataDefinition([-1, self.output_size], [torch.Tensor], "Batch of outputs [BATCH_SIZE x OUTPUT_SIZE]")
            }

    def forward(self, data_streams):
        """
        Main forward pass of the model.

        :param data_streams: DataStreams({'images',**})
        :type data_streams: ``ptp.dadatypes.DataStreams``
        """
        # Unpack DataStreams.
        enc_img = data_streams[self.key_image_encodings]
        enc_q = data_streams[self.key_question_encodings]

        sketch_pm_img = self.image_sketch_projection_matrix
        sketch_pm_q = self.question_sketch_projection_matrix

        # Project both batches.
        sketch_img = enc_img.mm(sketch_pm_img)
        sketch_q = enc_q.mm(sketch_pm_q)

        # Add imaginary parts (with zeros).
        sketch_img_reim = torch.stack([sketch_img, torch.zeros(sketch_img.shape).type(self.app_state.FloatTensor)], dim=2)
        sketch_q_reim = torch.stack([sketch_q, torch.zeros(sketch_q.shape).type(self.app_state.FloatTensor)], dim=2)
        #print("\n sketch_img_reim=",sketch_img_reim)
        #print("\n sketch_img_reim.shape=",sketch_img_reim.shape)

        # Perform FFT.
        # Returns the real and the imaginary parts together as one tensor of the same shape of input.
        fft_img = torch.fft(sketch_img_reim, signal_ndim=1)
        fft_q = torch.fft(sketch_q_reim, signal_ndim=1)
        #print(fft_img)

        # Get real and imaginary parts.
        real1 = fft_img[:,:,0]
        imag1 = fft_img[:,:,1]
        real2 = fft_q[:,:,0]
        imag2 = fft_q[:,:,1]

        # Calculate product.
        fft_product = torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim = -1)
        #print("fft_product=",fft_product)

        # Inverse FFT.
        cbp = torch.ifft(fft_product, signal_ndim=1)[:,:,0]
        #print("cbp=",cbp)

        # Add predictions to datadict.
        data_streams.publish({self.key_outputs: cbp})
