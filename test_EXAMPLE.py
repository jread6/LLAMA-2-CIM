import os
import sys
import torch
import torch.utils.data
from torch import nn
from torchvision import models

from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
from pytorch_quantization import cim

from inference import evaluate
from dataset import get_imagenet
from collections import namedtuple

# Definition of the named tuple that is used to store mapping of the quantized modules
_quant_entry = namedtuple('quant_entry', 'orig_mod mod_name replace_mod')

base_directory = 'example/' # set directory for accuracy results to be saved

# Global member of the file that contains the mapping of quantized modules
cim_quant_map = [_quant_entry(torch.nn, "Conv1d", quant_nn.QuantConv1d),
                _quant_entry(torch.nn, "Conv2d", cim.CIMConv2d),
                _quant_entry(torch.nn, "Conv3d", quant_nn.QuantConv3d),
                _quant_entry(torch.nn, "ConvTranspose1d", quant_nn.QuantConvTranspose1d),
                _quant_entry(torch.nn, "ConvTranspose2d", quant_nn.QuantConvTranspose2d),
                _quant_entry(torch.nn, "ConvTranspose3d", quant_nn.QuantConvTranspose3d),
                #TODO: replace linear laye with CIMLinear
                _quant_entry(torch.nn, "Linear", quant_nn.QuantLinear),
                _quant_entry(torch.nn, "LSTM", quant_nn.QuantLSTM),
                _quant_entry(torch.nn, "LSTMCell", quant_nn.QuantLSTMCell),
                _quant_entry(torch.nn, "AvgPool1d", quant_nn.QuantAvgPool1d),
                _quant_entry(torch.nn, "AvgPool2d", quant_nn.QuantAvgPool2d),
                _quant_entry(torch.nn, "AvgPool3d", quant_nn.QuantAvgPool3d),
                _quant_entry(torch.nn, "AdaptiveAvgPool1d", quant_nn.QuantAdaptiveAvgPool1d),
                _quant_entry(torch.nn, "AdaptiveAvgPool2d", quant_nn.QuantAdaptiveAvgPool2d),
                _quant_entry(torch.nn, "AdaptiveAvgPool3d", quant_nn.QuantAdaptiveAvgPool3d),]

def example():

    # initialize CIM simulation arguments
    device = 0  
    torch.cuda.set_device(device)   
    model_name = 'resnet18'
    LRS=1000
    state_1=2000
    state_2=4000
    HRS=1000000 #on/off = 1000

    mem_values=torch.tensor([HRS, LRS], device='cuda') # set memory state values
    res_std = torch.tensor([1000, 100], device='cuda')

    batch_size = 100    # batch size
    num_iter = 5        # number of batches to run inference on

    open_rows = 128
    adc_precision = 7 # adc precision can be reduced if open_rows is reduced
    # set current to voltage conversion method: e.g. trans-impedence amplifier (TIA) or pull-up PMOS (PU)
    I2V = 'TIA'       

    debug = False # set to true if you want additional debug messages  

    # set cim arguments
    cim_args = cim.CIMArgs(inference=True, 
                           batch_size=batch_size, 
                           mem_values=mem_values,
                           model_name=model_name,
                           hardware=False, # turn hardware simulation off for quantization calibration
                           adc_precision=adc_precision,
                           device=device)


    # use custom quant modules for cim compatible layers
    quant_modules.initialize(custom_quant_modules=cim_quant_map)

    iw_kwargs = {'fake_quant': True, 'unsigned': False, 'learn_amax' : False} # need to use fake quant because true quant is not supported by TensorRT, disable learn_amax because it has already been learned
    adc_kwargs = {'fake_quant': True, 'unsigned': False, 'learn_amax': True} 
    iw_quant_desc  = QuantDescriptor(calib_method='histogram', num_bits=8, **iw_kwargs)
    adc_quant_desc = QuantDescriptor(calib_method='histogram', num_bits=adc_precision, **adc_kwargs)

    cim.CIMConv2d.set_default_quant_desc_input(iw_quant_desc)
    cim.CIMConv2d.set_default_quant_desc_weight(iw_quant_desc)
    cim.CIMConv2d.set_default_quant_desc_adc(adc_quant_desc)
    cim.CIMConv2d.set_default_cim_args(cim_args)

    if model_name == 'resnet18':
        model = models.resnet18()
    elif model_name == 'resnet50':
        model = models.resnet50()
    elif model_name == 'swin_t':
        model = models.swin_v2_t()

    # some models have already been calibrated by James, they are stored in shimeng-srv2 at /usr/scratch1/datasets/
    # If you want to use a model not listed above, use calibrate.py to calibrate the model

    model_dir = '/usr/scratch1/james/models/'
    model_path = model_dir + '/cim_quant_' + model_name + '.pth'
    print("Loading quantized model...")
    model.load_state_dict(torch.load(model_path))
    print("Model loaded.")

    model.cuda()

    data_path = '/usr/scratch1/datasets/imagenet/'

    # use PyTorch's get_imagenet function
    data_loader, data_loader_test = get_imagenet(batch_size, data_path, train=True, val=True)

    filename = base_directory + 'results/accuracy/TEST_NAME.csv'

    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        print("Evaluating model...")
        evaluate(model, criterion, data_loader_test, device=device, print_freq=1, num_iter=num_iter, filename=filename)

if __name__ == '__main__':
    example()