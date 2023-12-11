#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


"""Some helper functions for implementing quantized modules"""
import copy
import inspect

from absl import logging

import torch
from torch import nn
import math
from copy import deepcopy

from pytorch_quantization.nn import TensorQuantizer
from pytorch_quantization.tensor_quant import QuantDescriptor, QUANT_DESC_8BIT_PER_TENSOR
from pytorch_quantization.cim.modules.args import CIMArgs


class QuantMixin():
    """Mixin class for adding basic quantization logic and cim parameters to quantized modules"""

    default_quant_desc_input      = QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_weight     = QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_adc        = QUANT_DESC_8BIT_PER_TENSOR

    @classmethod
    def set_default_quant_desc_input(cls, value):
        """
        Args:
            value: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`
        """
        if not isinstance(value, QuantDescriptor):
            raise ValueError("{} is not an instance of QuantDescriptor!")
        cls.default_quant_desc_input = copy.deepcopy(value)

    @classmethod
    def set_default_quant_desc_weight(cls, value):
        """
        Args:
            value: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`
        """
        if not isinstance(value, QuantDescriptor):
            raise ValueError("{} is not an instance of QuantDescriptor!")
        cls.default_quant_desc_weight = copy.deepcopy(value)

    @classmethod
    def set_default_quant_desc_adc(cls, value):
        """
        Args:
            value: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`
        """
        if not isinstance(value, QuantDescriptor):
            raise ValueError("{} is not an instance of QuantDescriptor!")
        cls.default_quant_desc_adc = copy.deepcopy(value)    

    @classmethod
    def set_default_cim_args(cls, value):
        """
        Args:
            value: An instance of :class:`CIMArgs <pytorch_quantization.cim.modules.args.CIMArgs>`
        """
        if not isinstance(value, CIMArgs):
            raise ValueError("{} is not an instance of CIMArgs!")
        cls.default_cim_args = copy.deepcopy(value)  

    def init_cim(self, cim_args, in_features, out_features):
        if not inspect.stack()[1].function == "__init__":
            raise TypeError("{} should be only called by __init__ of quantized module.".format(__name__))
        
        # TODO: Update CIMArgs to be a new class called CIM that where we construct cim_args from a cim_descriptor
        self._cim_args = deepcopy(cim_args)
        
        self._cim_args.weight2d_shape = [in_features, out_features]
        if self._cim_args.open_rows is None:
            self._cim_args.open_rows = self._cim_args.weight2d_shape[0]
        
        # NOTE: THIS DOES NOT SUPPORT MLC YET
        # calculate voltage references
        num_refs = (2**self._cim_args.adc_precision)
        x = torch.arange(num_refs) + 1

        # # subtract IR drop for this ADC block
        # vdd = self._cim_args.vdd - self._cim_args.logic_IR_drop

        vdd = self._cim_args.vdd
        LRS = self._cim_args.mem_values[-1]
        HRS = self._cim_args.mem_values[0]

        r_max = 1/(x/LRS)
        r_min = 1/((self._cim_args.open_rows-(x-1))/HRS + (x-1)/LRS)

        if self._cim_args.conversion_type == 'PU':
            print("ERROR: PU conversion type not supported yet")
            v_max = vdd*(r_max/(self._cim_args.res_divider + r_max))
            v_min = vdd*(r_min/(self._cim_args.res_divider + r_min))
            exit(1)

        elif self._cim_args.conversion_type == 'TIA':
            self._cim_args.Rf = LRS/self._cim_args.open_rows
            v_max = vdd*(self._cim_args.Rf/(r_min))
            v_min = vdd*(self._cim_args.Rf/(r_max))

        ###################################################################
        # v_max = torch.cat((torch.tensor([vdd], device=self._cim_args.device), v_max), 0)

        # self._cim_args.v_ref = (v_min[:-1]+v_max[1:])/2
        self._cim_args.v_ref = (v_min+v_max)/2
        ###################################################################


    def init_quantizer(self, quant_desc_input, quant_desc_weight, quant_desc_adc, num_layers=None, num_adc_quantizers=None):
        """Helper function for __init__ of quantized module

        Create input and weight quantizer based on quant_desc passed by kwargs, or default of the class.

        Args:
            quant_desc_input: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`
            quant_desc_weight: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`
            quant_desc_adc: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`            
            num_layers: An integer. Default None. If not None, create a list of quantizers.
        """
        if not inspect.stack()[1].function == "__init__":
            raise TypeError("{} should be only called by __init__ of quantized module.".format(__name__))
        self._fake_quant = True
        if (not quant_desc_input.fake_quant) or (not quant_desc_weight.fake_quant) or (not quant_desc_adc.fake_quant):
            raise ValueError("Only fake quantization is supported!")

        logging.info("Input is %squantized to %d bits in %s with axis %s!", ""
                     if not quant_desc_input.fake_quant else "fake ",
                     quant_desc_input.num_bits, self.__class__.__name__, quant_desc_input.axis)
        logging.info("Weight is %squantized to %d bits in %s with axis %s!", ""
                     if not quant_desc_weight.fake_quant else "fake ",
                     quant_desc_weight.num_bits, self.__class__.__name__, quant_desc_weight.axis)
        logging.info("Output is %squantized to %d bits in %s with axis %s!", ""
                     if not quant_desc_adc.fake_quant else "fake ",
                     quant_desc_adc.num_bits, self.__class__.__name__, quant_desc_adc.axis)

        if num_layers is None:
            self._input_quantizer = TensorQuantizer(quant_desc_input)
            self._weight_quantizer = TensorQuantizer(quant_desc_weight)
        else:
            self._input_quantizers = nn.ModuleList([TensorQuantizer(quant_desc_input) for _ in range(num_layers)])
            self._weight_quantizers = nn.ModuleList([TensorQuantizer(quant_desc_weight) for _ in range(num_layers)])

        if num_adc_quantizers is None:
            self._adc_quantizer = TensorQuantizer(quant_desc_adc)
        else:
            self._adc_quantizers = nn.ModuleList([TensorQuantizer(quant_desc_adc) for _ in range(num_adc_quantizers)])

    # pylint:disable=missing-docstring
    @property
    def input_quantizer(self):
        return self._input_quantizer

    @property
    def weight_quantizer(self):
        return self._weight_quantizer
    # pylint:enable=missing-docstring

    @property
    def adc_quantizer(self):
        return self._adc_quantizer


class QuantInputMixin():
    """Mixin class for adding basic quantization logic to quantized modules"""

    default_quant_desc_input = QUANT_DESC_8BIT_PER_TENSOR

    @classmethod
    def set_default_quant_desc_input(cls, value):
        """
        Args:
            value: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`
        """
        if not isinstance(value, QuantDescriptor):
            raise ValueError("{} is not an instance of QuantDescriptor!")
        cls.default_quant_desc_input = copy.deepcopy(value)

    def init_quantizer(self, quant_desc_input):
        """Helper function for __init__ of simple quantized module

        Create input quantizer based on quant_desc passed by kwargs, or default of the class.

        Args:
            quant_desc_input: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`
        """
        if not inspect.stack()[1].function == "__init__":
            raise TypeError("{} should be only called by __init__ of quantized module.".format(__name__))
        self._fake_quant = True
        if not quant_desc_input.fake_quant:
            raise ValueError("Only fake quantization is supported!")

        logging.info("Input is %squantized to %d bits in %s with axis %s!", ""
                     if not quant_desc_input.fake_quant else "fake ",
                     quant_desc_input.num_bits, self.__class__.__name__, quant_desc_input.axis)

        self._input_quantizer = TensorQuantizer(quant_desc_input)

    # pylint:disable=missing-docstring
    @property
    def input_quantizer(self):
        return self._input_quantizer
    # pylint:enable=missing-docstring


def pop_quant_desc_in_kwargs(quant_cls, input_only=False, **kwargs):
    """Pop quant descriptors in kwargs

    If there is no descriptor in kwargs, the default one in quant_cls will be used

    Arguments:
       quant_cls: A class that has default quantization descriptors
       input_only: A boolean. If True, pop quant_desc_input only, not quant_desc_weight. Default false.

    Keyword Arguments:
       quant_desc_input: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
           Quantization descriptor of input.
       quant_desc_weight: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
           Quantization descriptor of weight.
    """
    quant_desc_input = kwargs.pop('quant_desc_input', quant_cls.default_quant_desc_input)
    if not input_only:
        quant_desc_weight = kwargs.pop('quant_desc_weight', quant_cls.default_quant_desc_weight)

    quant_desc_adc = kwargs.pop('quant_desc_adc', quant_cls.default_quant_desc_adc)

    cim_args = kwargs.pop('cim_args', quant_cls.default_cim_args)

    # Check if anything is left in **kwargs
    if kwargs:
        raise TypeError("Unused keys: {}".format(kwargs.keys()))

    if input_only:
        return quant_desc_input
    return quant_desc_input, quant_desc_weight, quant_desc_adc, cim_args
