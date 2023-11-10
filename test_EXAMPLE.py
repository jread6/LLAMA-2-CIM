import os
import sys
import torch
import torch.utils.data
from torch import nn
from torchvision import models

from pytorch_quantization import nn as quant_nn


from inference import evaluate
from dataset import get_imagenet

from pytorch_quantization import quant_modules

sys.path.append("/usr/scratch1/james/DNN_NeuroSim_V1.5/pytorch_quantization/cim") # change this file location to your local installation of pytorch-quantization
from args import CIMArgs

base_directory = 'example/' # set directory for accuracy results to be saved

def example():

    # initialize CIM simulation arguments
    torch.cuda.set_device(1)   
    device = 'cuda'    
    model_name = 'ResNet18'
    LRS=1000
    state_1=2000
    state_2=4000
    HRS=1000000 #on/off = 1000

    mem_values=torch.tensor([HRS, state_2, state_1, LRS], device='cuda') # set memory state values
    res_std = torch.tensor([1000, 400, 200, 100], device='cuda')

    batch_size = 100    # batch size
    num_iter = 5        # number of batches to run inference on

    open_rows = 128
    adc_precision = 7 # adc precision can be reduced if open_rows is reduced
    # set current to voltage conversion method: e.g. trans-impedence amplifier (TIA) or pull-up PMOS (PU)
    I2V = 'TIA'       

    debug = False # set to true if you want additional debug messages  

    # set cim arguments
    cim_args = CIMArgs(inference=True, adc_precision=adc_precision, 
                    open_rows=open_rows, batch_size=batch_size,
                    mem_values=mem_values, resistance_std=res_std, conversion_type=I2V,
                    debug=debug, model_name=model_name,
                    calc_BER=False, device=device)


    # set cim arguments for quantized layers in TensorRT
    quant_nn.QuantConv2d.set_cim_args(cim_args)
    quant_nn.QuantLinear.set_cim_args(cim_args)

    quant_modules.initialize()

    if model_name == 'ResNet18':
        model = models.resnet18()
    elif model_name == 'ResNet50':
        model = models.resnet50()
    elif model_name == 'swin_t':
        model = models.swin_v2_t()

    # some models have already been calibrated by James, they are stored in shimeng-srv2 at /usr/scratch1/datasets/
    # If you want to use a model not listed above, use calibrate.py to calibrate the model
    print("Loading quantized model...")
    model.load_state_dict(torch.load("/usr/scratch1/datasets/quant_"+model_name+"-calibrated.pth"))
    print("Model loaded.")

    model.cuda()

    data_path = '/usr/scratch1/datasets/imagenet/'

    # use PyTorch's get_imagenet function
    data_loader, data_loader_test = get_imagenet(batch_size, data_path, train=True, val=True)

    filename = base_directory + 'results/accuracy/TEST_NAME.csv'

    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        print("Evaluating model...")
        evaluate(model, criterion, data_loader_test, device="cuda", print_freq=1, num_iter=num_iter, filename=filename)

if __name__ == '__main__':
    example()