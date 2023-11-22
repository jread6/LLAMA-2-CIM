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
import math

# import cim
# from cim.tensor_quant import QuantDescriptor
# import cim_modules
# cim_modules.initialize()

import pytorch_quantization.cim.modules.macro as macro
import pytorch_quantization.nn as quant_nn
import pytorch_quantization.calib as calib
from pytorch_quantization.tensor_quant import QuantDescriptor
import pytorch_quantization.quant_modules as quant_modules
from pytorch_quantization import cim

from dataset import get_imagenet
from inference import evaluate
from collections import namedtuple

from plot_hist import plot_hist

# Definition of the named tuple that is used to store mapping of the quantized modules
_quant_entry = namedtuple('quant_entry', 'orig_mod mod_name replace_mod')

# Global member of the file that contains the mapping of quantized modules
cim_quant_map = [_quant_entry(torch.nn, "Conv2d", cim.CIMConv2d),]


# Global member of the file that contains the mapping of quantized modules
quant_map = [_quant_entry(torch.nn, "Conv1d", quant_nn.QuantConv1d),
                _quant_entry(torch.nn, "Conv2d", quant_nn.QuantConv2d),
                _quant_entry(torch.nn, "Conv3d", quant_nn.QuantConv3d),
                _quant_entry(torch.nn, "ConvTranspose1d", quant_nn.QuantConvTranspose1d),
                _quant_entry(torch.nn, "ConvTranspose2d", quant_nn.QuantConvTranspose2d),
                _quant_entry(torch.nn, "ConvTranspose3d", quant_nn.QuantConvTranspose3d),
                _quant_entry(torch.nn, "Linear", quant_nn.QuantLinear),
                _quant_entry(torch.nn, "LSTM", quant_nn.QuantLSTM),
                _quant_entry(torch.nn, "LSTMCell", quant_nn.QuantLSTMCell),
                _quant_entry(torch.nn, "AvgPool1d", quant_nn.QuantAvgPool1d),
                _quant_entry(torch.nn, "AvgPool2d", quant_nn.QuantAvgPool2d),
                _quant_entry(torch.nn, "AvgPool3d", quant_nn.QuantAvgPool3d),
                _quant_entry(torch.nn, "AdaptiveAvgPool1d", quant_nn.QuantAdaptiveAvgPool1d),
                _quant_entry(torch.nn, "AdaptiveAvgPool2d", quant_nn.QuantAdaptiveAvgPool2d),
                _quant_entry(torch.nn, "AdaptiveAvgPool3d", quant_nn.QuantAdaptiveAvgPool3d),]

def main(model_name='resnet18', dataset_name='imagenet'):
    gpu = 0
    batch_size = 128
    torch.cuda.set_device(gpu)
    torch.manual_seed(0)
    
    HRS = 1000000
    LRS = 1000
    mem_values = torch.tensor([HRS, LRS], device=gpu)
    # initialize CIM simulation arguments
    cim_args = cim.CIMArgs(inference=True, 
                           batch_size=batch_size, 
                           mem_values=mem_values,
                           model_name=model_name,
                           open_rows=128,
                           adc_reduction=0,
                           hardware=False, # turn hardware simulation off for quantization calibration
                           device=gpu)

    # use custom quant modules for cim compatible layers
    quant_modules.initialize(float_module_list=['Conv2d'], custom_quant_modules=cim_quant_map)

    iw_kwargs  = {'fake_quant': True, 'unsigned': False} # need to use fake quant because true quant is not supported by TensorRT, disable learn_amax because it has already been learned
    adc_kwargs = {'fake_quant': True, 'unsigned': True} 
    iw_quant_desc  = QuantDescriptor(calib_method='histogram', num_bits=8, **iw_kwargs)
    adc_quant_desc = QuantDescriptor(calib_method='histogram', **adc_kwargs)

    cim.CIMConv2d.set_default_quant_desc_input(iw_quant_desc)
    cim.CIMConv2d.set_default_quant_desc_weight(iw_quant_desc)
    cim.CIMConv2d.set_default_quant_desc_adc(adc_quant_desc)
    cim.CIMConv2d.set_default_cim_args(cim_args)

    # TODO: support CIMLinear
    # cim.CIMLinear.set_default_quant_desc_input(iw_quant_desc)
    # cim.CIMLinear.set_default_quant_desc_weight(iw_quant_desc)
    # cim.CIMLinear.set_default_quant_desc_adc(adc_quant_desc)
    # cim.CIMLinear.set_default_cim_args(cim_args)


    # quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc)
    # quant_nn.QuantConv2d.set_default_quant_desc_weight(quant_desc)

    # quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc)
    # quant_nn.QuantLinear.set_default_quant_desc_weight(quant_desc)
    model_dir = '/usr/scratch1/james/models/'
    data_path = '/usr/scratch1/datasets/imagenet/'

    if model_name == 'resnet18':
        model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
    elif model_name == 'resnet50':
        model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    elif model_name == 'swin_t':
        model = models.swin_v2_t(weights='DEFAULT')

    model.cuda()
    data_loader, data_loader_test = get_imagenet(batch_size, data_path, train=True, val=True, sample=False)

    try:
        # load model
        print("Loading input and weight quantized model...")
        iw_quant_model = '/usr/scratch1/james/models/iw_quant_' + model_name + '.pth'
        model.load_state_dict(torch.load(iw_quant_model))
    except:
        # step 1: quantize inputs and weights with TensorRT
        print("Quantizing inputs and weights...")
        # Collect statistics
        # It is a bit slow since we collect histograms on CPU
        with torch.no_grad():
            collect_stats(model, data_loader, num_batches=2, cim=False)
            strict = True
            if model_name == 'swin_t':
                strict = False
            compute_amax(model, quant='iw', method="percentile", percentile=99.99, strict=strict)

    num_iter=1

    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        print('Evaluating iw quantized model...')
        baseline_accuracy = evaluate(model, criterion, data_loader_test, device=gpu, num_iter=num_iter, print_freq=1)

        # Save the model
        calibrated_model = model_dir + '/iw_quant_' + model_name + '.pth'
        torch.save(model.state_dict(), calibrated_model)


        # create a list of all layers
        layers = []
        for name, module in model.named_modules():
            if isinstance(module, macro.CIM):
                layers.append(name+'._adc_quantizer')
                module._cim_args.quant_mode = 'adc'

        # pick which layers to quantize the adc outputs
        layer_quant = layers
        # layer_quant = [layers[3], layers[4]]
        # layer_quant = [layers[4]]
        # TODO: log all layers and their ideal adc_precision

        try:
            # load model
            print("Loading adc quantized model...")
            adc_quant_model = model_dir + '/cim_quant_' + model_name + '.pth'
            model.load_state_dict(torch.load(adc_quant_model))
        except:
            # step 2: quantize adc outputs with TensorRT
            print("Gathering ADC output histograms...")
            # Collect statistics
            # It is a bit slow since we collect histograms on CPU

            collect_stats(model, data_loader, num_batches=2, quant_mode='adc', layer_quant=layer_quant)
            strict = True
            if model_name == 'swin_t':
                strict = False
            compute_amax(model, quant_mode='adc', layer_quant=layer_quant, method="percentile", percentile=99.99, strict=strict)

        calibrated_model = model_dir + '/cim_quant_' + model_name + '.pth'
        torch.save(model.state_dict(), calibrated_model)

        print("Optimizing ADC precision for each layer...")
        optimize_adc_precision(model, data_loader_test, layer_quant, baseline_accuracy, device=gpu) 

    print(model.state_dict().keys())
    calibrated_model = model_dir + '/cim_quant_' + model_name + '.pth'
    torch.save(model.state_dict(), calibrated_model)

    for name, module in model.named_modules():
        if isinstance(module, cim.CIMConv2d):
            print(f'{name} ADC precision: {module._cim_args.adc_precision}')

def collect_stats(model, data_loader, num_batches, quant_mode='iw', layer_quant=[1]):
    """Feed data to the network and collect statistic"""

    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if quant_mode == 'iw' and 'adc' not in name and module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            
            elif quant_mode == 'adc' and 'adc' in name and module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
   

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

def compute_amax(model, quant_mode='iw', layer_quant=[], **kwargs):
    amax = 0
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if quant_mode == 'iw' and 'adc' not in name:
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax(strict=False)
                    else:
                        module.load_calib_amax(**kwargs)

            elif quant_mode == 'adc' and name in layer_quant:
                if module._calibrator is not None and module._disabled == False:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax(strict=False)
                    else:
                        module.load_calib_amax(**kwargs)

                    # plot_hist(name=name, calib=module._calibrator, percentile=99.99)
                    print(F"{name:40}: {module}")


        # if isinstance(module, macro.CIM):
            # turn off calibration mode
            # module._cim_args.quant_mode = None
        #     #TODO: log adc precision for this layer
        #     module._cim_args.adc_precision = math.ceil(math.log2(amax))
            
    model.cuda()

def optimize_adc_precision(model, data_loader_test, layer_quant, baseline_accuracy, accuracy_tolerance=3.5, device=0):

    num_iter=1

    # disable all adc quantizers
    for name, module in model.named_modules():
        if isinstance(module, macro.CIM):  
            module._adc_quantizer.disable()

    # start with the last layer
    layer_quant.reverse()

    # determine minimum adc precision for each layer
    for layer in layer_quant:
        for name, module in model.named_modules():
            if isinstance(module, macro.CIM) and name+'._adc_quantizer' == layer:  
                # enable this quantizer
                module._adc_quantizer.enable()             
                accuracy = 0
                amax = module._adc_quantizer.amax
                adc_precision = math.ceil(math.log2(amax)) - 1
                while accuracy < baseline_accuracy - accuracy_tolerance and adc_precision <= module._cim_args.ideal_adc_precision:
                    
                    module._cim_args.adc_precision = adc_precision

                    criterion = nn.CrossEntropyLoss()
                    with torch.no_grad():
                        print(f'\nTesting {name} (amax: {amax}) with {adc_precision}b adc precision...')
                        accuracy = evaluate(model, criterion, data_loader_test, device=device, num_iter=num_iter, print_freq=1)

                    adc_precision += 1
            

if __name__ == '__main__':
    main()