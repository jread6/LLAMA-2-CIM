# import datetime
# import os
# import time
import warnings

# import presets
import torch
import torch.utils.data
import torch.nn as nn
from tqdm import tqdm
# import torchvision
# import transforms
from torchvision import models
from pytorch_quantization import cim
import pytorch_quantization.cim.modules.macro as macro

from dataset import get_imagenet
from collections import namedtuple

# import vision.references.classification.utils as utils
# from sampler import RASampler
# from torch import nn
# from torch.utils.data.dataloader import default_collate
# from torchvision.transforms.functional import InterpolationMode
import sys
# sys.path.append('pytorch-quantization/')

import pytorch_quantization.quant_modules as quant_modules

# Definition of the named tuple that is used to store mapping of the quantized modules
_quant_entry = namedtuple('quant_entry', 'orig_mod mod_name replace_mod')

# Global member of the file that contains the mapping of quantized modules
cim_quant_map = [_quant_entry(torch.nn, "Conv2d", cim.CIMConv2d),]

def main(model_name='resnet18'):
    gpu = 0
    batch_size = 128
    HRS = 1000000
    LRS = 1000
    mem_values = torch.tensor([HRS, LRS], device=gpu)
    torch.cuda.set_device(gpu)
    torch.manual_seed(0)
    
    # initialize CIM simulation arguments
    cim_args = cim.CIMArgs(inference=True, 
                           batch_size=batch_size, 
                           mem_values=mem_values,
                           model_name=model_name,
                           open_rows=128,
                           adc_reduction=0,
                           hardware=False, # turn hardware simulation off for quantization calibration
                           quant_mode='adc',
                           device=gpu)

    # use custom quant modules for cim compatible layers
    quant_modules.initialize(float_module_list=['Conv2d'], custom_quant_modules=cim_quant_map)
    cim.CIMConv2d.set_default_cim_args(cim_args)

    if model_name == 'resnet18':
        model = models.resnet18()
    elif model_name == 'resnet50':
        model = models.resnet50()
    elif model_name == 'swin_t':
        model = models.swin_v2_t()

    # some models have already been calibrated by James, they are stored in shimeng-srv2 at /usr/scratch1/datasets/
    # If you want to use a model not listed above, use calibrate.py to calibrate the model

    try:
        print("Loading quantized model...")
        state_dict = torch.load("/usr/scratch1/james/models/cim_quant_"+model_name+".pth")
        model.load_state_dict(state_dict)
        print("Model loaded.")
    except:
        print("Quantized model not found, please quantize the model first.")
        return

    model.cuda()
    for name, module in model.named_modules():
        if isinstance(module, macro.CIM):
            print(f'{name} ADC precision: {module._cim_args.adc_precision}')

    data_path = '/usr/scratch1/datasets/imagenet/'
    num_iter = 1

    # use PyTorch's get_imagenet function
    data_loader_test = get_imagenet(batch_size, data_path, train=False, val=True)

    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        print("Evaluating model...")
        evaluate(model, criterion, data_loader_test, device="cuda", print_freq=1, num_iter=num_iter)

def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix="", num_iter=5, filename=None):
        
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        total_loss = 0
        test_count = 0
        for i, (images, labels) in enumerate(data_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            test_count += 1

            if test_count == num_iter:
                total_loss /= test_count

                print('Test Accuracy of the model on {} test images: {} %'.format(num_iter*images.shape[0], 100 * correct / total))
                print('Average Test loss: {}'.format(total_loss))

                return 100 * correct / total

        total_loss /= test_count

        print('Test Accuracy of the model on the 50000 test images: {} %'.format(100 * correct / total))
        print('Average Test loss: {}'.format(total_loss))

if __name__ == '__main__':
    main()