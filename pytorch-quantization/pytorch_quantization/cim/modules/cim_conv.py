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


"""Quantized convolution
Base code is from nn.Conv, details of Module and original argument can be found there.
Module names are intentionally kept same as unquantized version so that they can be dropped into preexisting model
easily, and load pretrained weight. Aliases with Quant prefix are defined and are encouraged to be used explicitly
when start scratch.
"""

import inspect
import torch
import math
import torch.nn
import torch.nn.functional as F
from copy import deepcopy
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.modules.conv import _ConvTransposeNd
from pytorch_quantization.tensor_quant import QuantDescriptor

from pytorch_quantization import tensor_quant
import pytorch_quantization.cim.modules.macro as macro
import pytorch_quantization.cim.modules.args as args

from pytorch_quantization.nn.modules import _utils

import pytorch_quantization.cim.modules._utils as _cim_utils

__all__ = [
    "Conv2d", "CIMConv2d"
]


class _CIMConvNd(torch.nn.modules.conv._ConvNd, macro.CIM, _cim_utils.QuantMixin):
    """base class of quantized CIMConv inherited from _ConvNd

    Comments of original arguments can be found in torch.nn.modules.conv

    Arguments:
        quant_desc_input: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
            Quantization descriptor of input.
        quant_desc_weight: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
            Quantization descriptor of weight.
        quant_desc_ADCoutput: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
            Quantization descriptor of ADC outputs.

    Raises:
        ValueError: If unsupported arguments are passed in.

    Readonly properties:
        - input_quantizer:
        - weight_quantizer:
        - output_quantizer:

    Static methods:
        - set_default_quant_desc_input: Set default_quant_desc_input
        - set_default_quant_desc_weight: Set default_quant_desc_weight
    """

    default_quant_desc_input  = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_adc    = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    default_cim_args          = args.CIMArgs()

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode, quant_desc_input, quant_desc_weight, quant_desc_adc, cim_args):
        
        super(_CIMConvNd, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                           transposed, output_padding, groups, bias, padding_mode)

        _cim_args = deepcopy(cim_args)

        _cim_args.weight2d_shape = [kernel_size[0]*kernel_size[1]*in_channels, out_channels]
        _cim_args.ideal_adc_precision = math.ceil(math.log2(min(_cim_args.open_rows, _cim_args.weight2d_shape[0])))
        self.init_cim(_cim_args)

        # allow for different adc quantization for different layers (per tile)
        num_rows = min(self._cim_args.open_rows, self._cim_args.weight2d_shape[0])
        ideal_precision = math.ceil(math.log2(num_rows))

        adc_quant_desc = QuantDescriptor(calib_method='histogram', num_bits=ideal_precision, unsigned=True)
        self.init_quantizer(quant_desc_input, quant_desc_weight, adc_quant_desc)

    def _quant(self, input):
        """Apply quantization on input and weight

        Function called by the classes lower in the hierarchy, which actually performs the quantization before forward
        in the derivate class the particular Function.

        Arguments:
            input: in_features to quantize
        Returns:
            A tuple: (quant_in_feature, quant_weight)
        """
        quant_input  = self._input_quantizer(input)
        quant_weight = self._weight_quantizer(self.weight)

        # after quantization of inputs and weights, convert fake quantized input and weight to integers
        if self._cim_args.quant_mode is None or self._cim_args.quant_mode == 'adc':
            input_bits     = self.input_quantizer.num_bits
            input_unsigned = self.input_quantizer.unsigned
            input_amax     = self.input_quantizer.amax

            input_max_bound = torch.tensor((2.0**(input_bits - 1 + int(input_unsigned))) - 1.0, device=quant_input.device)
            scale = input_max_bound / input_amax
            self.input_scale = scale
            quant_input = quant_input*scale

            weight_bits = self.weight_quantizer.num_bits
            weight_unsigned = self.weight_quantizer.unsigned
            weight_amax     = self.weight_quantizer.amax

            weight_max_bound = torch.tensor((2.0**(weight_bits - 1 + int(weight_unsigned))) - 1.0, device=quant_weight.device)
            scale = weight_max_bound / weight_amax
            self.weight_scale = scale
            quant_weight = quant_weight*scale

        return (quant_input, quant_weight)
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state = super(_CIMConvNd, self).state_dict(destination, prefix, keep_vars)
        state[prefix + '_cim_args'] = self._cim_args
        return state
    
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        self._cim_args = state_dict[prefix + '_cim_args']
        del state_dict[prefix + '_cim_args']
        super(_CIMConvNd, self)._load_from_state_dict(state_dict, prefix, *args, **kwargs)

class CIMConv2d(_CIMConvNd):
    """Quantized 2D conv"""

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 **kwargs):

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        quant_desc_input, quant_desc_weight, quant_desc_adc, cim_args = _cim_utils.pop_quant_desc_in_kwargs(self.__class__, **kwargs)

        super(CIMConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False,
                                          _pair(0), groups, bias, padding_mode,
                                          quant_desc_input=quant_desc_input, quant_desc_weight=quant_desc_weight, 
                                          quant_desc_adc=quant_desc_adc, cim_args=cim_args)

    def forward(self, input):
        # the actual quantization happens in the next level of the class hierarchy
        quant_input, quant_weight = self._quant(input)
        # if we are quantizing the weights and inputs, don't de-quantize

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            output = self.conv2d(F.pad(quant_input, expanded_padding, mode='circular'),
                              quant_weight, self.bias, self.stride,
                              _pair(0), self.dilation, self.groups)
        else:
            output = self.conv2d(quant_input, quant_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        # print(input)
        # print(output)

        return output

    def conv2d(self, input, weight, bias, stride, padding, dilation, groups):
        """Forward function of CIMConv2d"""

        # if we are still calibrating input and weight quantization, use the original convolution function
        if self._cim_args.quant_mode == 'iw':
            return F.conv2d(input, weight, bias, stride, padding, dilation, groups)
        
        #TODO: support circular convolution

        # Unfold the input tensor into a 2D matrix
        # TODO: Use Novel mapping method to reduce memory consumption
        
        input_2d = F.unfold(input, kernel_size=weight.shape[-2:], dilation=dilation,
                                padding=padding, stride=stride).transpose(1, 2).flatten(start_dim=0, end_dim=1)
        
        input  = input.to(torch.int32)
        weight = weight.to(torch.int32)
        
        # Reshape the weight tensor into a 2D matrix
        # TODO: check shape: num_filters, num_channels, kernel_size, kernel_size
        weight_2d = weight.view(weight.shape[0], -1).t()

        # simulate_array is the same for any convolution and is defined in macro.py
        output = self.simulate_array(input_2d, weight_2d)

        # Reshape the 2D output matrix into a 4D tensor
        N = input.shape[0]
        C_out = weight.shape[0]
        H_out = (input.shape[-2] + 2 * self.padding[0] - self.dilation[0] * (weight.shape[-2] - 1) - 1) // self.stride[0] + 1
        W_out = (input.shape[-1] + 2 * self.padding[1] - self.dilation[1] * (weight.shape[-1] - 1) - 1) // self.stride[1] + 1
        output = output.view(N, H_out, W_out, C_out).permute(0, 3, 1, 2)

        # Add bias if provided
        if bias is not None:
            output += bias.view(1, -1, 1, 1)

        # de-quantize integer outputs back to floating point
        scale = self.weight_scale * self.input_scale

        return output/scale


# Define alias with Quant prefix
_ConvNd = _CIMConvNd
Conv2d = CIMConv2d

