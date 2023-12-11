#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


"""Quantized Linear"""
import torch
from torch import nn
from torch.nn import functional as F
import math
from copy import deepcopy

from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import tensor_quant
import pytorch_quantization.cim.modules.macro as macro
import pytorch_quantization.cim.modules.args as args
import pytorch_quantization.cim.modules._utils as _cim_utils


from . import _utils

__all__ = ["Linear", "CIMLinear"]

class CIMLinear(nn.modules.linear.Linear, macro.CIM, _utils.QuantMixin):
    """Quantized version of nn.Linear

    Apply quantized linear to the incoming data, y = dequant(quant(x)quant(A)^T + b).

    Keep Module name "Linear" instead of "CIMLinear" so that it can be easily dropped into preexisting model and load
    pretrained weights. An alias "CIMLinear" is defined below. The base code is a copy of nn.Linear, see detailed
    comment of original arguments there.

    Quantization descriptors are passed in in kwargs. If not presents, default_quant_desc_input and
    default_quant_desc_weight are used.

    Keyword Arguments:
        quant_desc_input: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
            Quantization descriptor of input.
        quant_desc_wegiht: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
            Quantization descriptor of weight.

    Raises:
        ValueError: If unsupported arguments are passed in.
        KeyError: If unsupported kwargs are passed in.

    Readonly properties:
        - input_quantizer:
        - weight_quantizer:

    Static methods:
        - set_default_quant_desc_input: Set default_quant_desc_input
        - set_default_quant_desc_weight: Set default_quant_desc_weight
    """

    default_quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW
    default_quant_desc_adc    = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    default_cim_args          = args.CIMArgs()

    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super(CIMLinear, self).__init__(in_features, out_features, bias)
        
        quant_desc_input, quant_desc_weight, quant_desc_adc, cim_args = _cim_utils.pop_quant_desc_in_kwargs(self.__class__, **kwargs)
        
        self.init_quantizer(quant_desc_input, quant_desc_weight, quant_desc_adc)
        # TODO: update convolution to match this
        self.init_cim(cim_args, in_features, out_features) 

    def _to_int(self, input):
        """

        Converts the quantized input and weight into their integer representations

        Arguments:
            input: in_features to quantize
        Returns:
            A tuple: (quant_in_feature, quant_weight)
        """

        # after quantization of inputs and weights, convert fake quantized input and weight to integers
        quant_input  = self._input_quantizer(input)
        quant_weight = self._weight_quantizer(self.weight)

        # TODO: wrap this section in a function
        input_bits     = self.input_quantizer.num_bits
        input_unsigned = self.input_quantizer.unsigned
        input_amax     = self.input_quantizer.amax

        input_max_bound = torch.tensor((2.0**(input_bits - 1 + int(input_unsigned))) - 1.0, device=input_amax.device)
        scale = (input_max_bound / input_amax).to(quant_input.device)
        # scale = (input_max_bound / input_amax)
        self.input_scale = scale

        # print(f'input_scale device: {scale.device}\n')
        # print(f'quant_input device: {quant_input.device}\n')

        quant_input = quant_input*scale

        ############################################################

        weight_bits = self.weight_quantizer.num_bits
        weight_unsigned = self.weight_quantizer.unsigned
        weight_amax     = self.weight_quantizer.amax

        weight_max_bound = torch.tensor((2.0**(weight_bits - 1 + int(weight_unsigned))) - 1.0, device=weight_amax.device)
        scale = (weight_max_bound / weight_amax).to(quant_weight.device)
        # scale = (weight_max_bound / weight_amax)
        self.weight_scale = scale

        # print(f'weight_scale device: {scale.device}\n')
        # print(f'quant_weight device: {quant_weight.device}\n')

        quant_weight = quant_weight*scale

        return (quant_input, quant_weight)
    
    def forward(self, input):

        # the actual quantization happens in the next level of the class hierarchy

        if self._cim_args.quant_mode == 'iw' or self._input_quantizer._disabled or self._weight_quantizer._disabled or self._adc_quantizer._disabled:
            quant_input  = self._input_quantizer(input)
            quant_weight = self._weight_quantizer(self.weight)
            output = F.linear(quant_input, quant_weight, bias=self.bias)

            if self.bias is not None:
                output += self.bias

            return output

        else:        
            quant_input, quant_weight = self._to_int(input)

            # print(f'input device: {input.device}\n')
            # print(f'quant_input device: {quant_input.device}\n')
            # print(f'quant_weight device: {quant_weight.device}\n')

            input_shape = quant_input.shape
            weight_shape = quant_weight.shape

            # Reshape the input into a 2D matrix
            if len(input_shape) > 2:
                quant_input = quant_input.flatten(start_dim=0, end_dim=-2)

            # Reshape the weight tensor into a matrix
            quant_weight = quant_weight.t()

            # Perform matrix multiplication with CIM
            output = self.simulate_array(quant_input, quant_weight)

            # Reshape output
            if len(input_shape) > 2:
                output_shape = input_shape[:-1] + weight_shape[0:1] # need 0:1 to keep it as a tuple
                output = output.reshape(output_shape)

            # De-quantize output
            # print(f'input shape: {input_shape}\n')
            # print(f'input scale shape: {self.input_scale.shape}\n')
            # print(f'weight shape: {weight_shape}')
            # print(f'weight_scale shape: {self.weight_scale.shape}\n')
            # print(f'output shape: {output.shape}\n')
            scale = self.input_scale * self.weight_scale.t()
            output = output/scale

            # Add bias if provided
            # TODO: do this with CIM ? 
            if self.bias is not None:
                output += self.bias

            return output


Linear = CIMLinear
