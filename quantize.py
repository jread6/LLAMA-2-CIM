# step 1: quantize the input and weights of the model
# step 2: quantize the ADC outputs

# the QuantConv class inherets from both nn.ConvNd and QuantMixin
# class _QuantConvNd(torch.nn.modules.conv._ConvNd, _utils.QuantMixin):
# QuantMixin has two parameters: input_quantizer and weight_quantizer which are TensorQuantizer modules
# TensorQuantizer inherets from nn.Module

# QuantConv calls the _quant() function to quantize the inputs and the weights based on their calculated amax values
# _quant calls the input and weight quantizers that are the member of QuantConv

# CIMConv:
    # 1. call _quant to quantize the inputs and weights
    # 2. call simulate_array to perform the unfolded convolution
        # a. calculate column outputs with simulate array
            # aa. if we are in the calibration stage, call _adc_quant() on the column outputs
            # ab. simulate_array() will need it's own adc output quantizer (TensorQuantizer) 
            # ac. after we have calculated the amax values for the column outputs during the calibration, use amax to determine the adc precision
        # b. if we are not in calibration stage, call adc() on the column outputs


# monkey patch all instances of Conv2d for these classes depending on user preference

# ptq_cim.Conv2d:
# - use TensorRT to quantize activations and weights
# - use TensorRT to calibrate the quantization of ADC outputs
# - test inference accuracy under cim noises

# ft_cim.Conv2d:
# - fine tune ptq model with cim noises

# qat_cim.Conv2d: 
# - user other quantization framework for qat + hat


import os
import sys
import torch
import torch.utils.data
from torch import nn
from tqdm import tqdm
from torchvision import models

# import cim
# from cim.tensor_quant import QuantDescriptor
# import cim_modules
# cim_modules.initialize()

import pytorch_quantization.nn as quant_nn
import pytorch_quantization.calib as calib
from pytorch_quantization.tensor_quant import QuantDescriptor
import pytorch_quantization.quant_modules as quant_modules
quant_modules.initialize()

from dataset import get_imagenet
from inference import evaluate


# IDEA: change name of pytorch-quantization
# OR: just copy pytorch_quantization into NeuroSim repo

def main(model_name='resnet18', dataset_name='imagenet'):
    torch.cuda.set_device(1)

    # TODO: CHANGE MODULE MAPPING

    kwargs = {'fake_quant': True, 'unsigned': False}
    quant_desc = QuantDescriptor(calib_method='histogram', **kwargs)
    print(quant_desc)

    # cim.CIMConv2d.set_default_quant_desc_input(quant_desc_input)
    # cim.CIMLinear.set_default_quant_desc_input(quant_desc_input)

    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc)
    quant_nn.QuantConv2d.set_default_quant_desc_weight(quant_desc)

    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc)
    quant_nn.QuantLinear.set_default_quant_desc_weight(quant_desc)

    if model_name == 'resnet50':
        model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    elif model_name == 'resnet18':
        model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
    elif model_name == 'swin_t':
        model = models.swin_v2_t(weights='DEFAULT')

    model.cuda()

    data_path = '/usr/scratch1/datasets/imagenet/'
    batch_size = 512

    # traindir = os.path.join(data_path, 'train')
    # valdir = os.path.join(data_path, 'val')

    data_loader, data_loader_test = get_imagenet(batch_size, data_path, train=True, val=True, sample=True)

    # Collect statistics
    # It is a bit slow since we collect histograms on CPU
    with torch.no_grad():
        collect_stats(model, data_loader, num_batches=2)
        strict = True
        if model_name == 'swin_t':
            strict = False
        compute_amax(model, method="percentile", percentile=99.99, strict=strict)

    print(model)
    num_iter=10

    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        evaluate(model, criterion, data_loader_test, device="cuda", num_iter=num_iter, print_freq=20)

    # Save the model
    calibrated_model = '/usr/scratch1/james/models/quant_' + model_name + '-calibrated.pth'
    torch.save(model.state_dict(), calibrated_model)



def collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistic"""

    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
        model(image.cuda())
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax(strict=False)
                else:
                    module.load_calib_amax(**kwargs)
            print(F"{name:40}: {module}")
    model.cuda()

if __name__ == '__main__':
    main()