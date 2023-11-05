import os
import sys
import torch
import torch.utils.data
from torch import nn
from tqdm import tqdm
from torchvision import models

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor


sys.path.append("/usr/scratch1/james/vision/references/classification")
sys.path.append("/usr/scratch1/james/Inference_NeuroSim/Inference_pytorch/models/")
from train import evaluate, train_one_epoch
from dataset import get_imagenet

from pytorch_quantization import quant_modules
quant_modules.initialize()

# if torch.distributed.is_available():
#     torch.distributed.init_process_group(backend='nccl', world_size=4, rank=0)

def main(model_name='ResNet50', dataset_name='imagenet'):
    torch.cuda.set_device(1)  

    quant_desc_input = QuantDescriptor(calib_method='histogram')
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
    
    if model_name == 'ResNet50':
        model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    elif model_name == 'ResNet18':
        model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
    if model_name == 'ResNet101':
        model = models.resnet101(weights='ResNet101_Weights.DEFAULT')
    if model_name == 'ResNet152':
        model = models.resnet152(weights='ResNet152_Weights.DEFAULT')
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

    model.cuda()

    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        evaluate(model, criterion, data_loader_test, device="cuda", print_freq=20)

    # Save the model
    calibrated_model = '/usr/scratch1/datasets/quant_' + model_name + '-calibrated.pth'
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
                    # module.load_calib_amax(strict=False)
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(F"{name:40}: {module}")
    model.cuda()

if __name__ == "__main__":
    main('ResNet18')