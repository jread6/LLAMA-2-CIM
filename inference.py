# import datetime
# import os
# import time
import warnings

# import presets
import torch
import torch.utils.data
# import torchvision
# import transforms
import torchvision
# import vision.references.classification.utils as utils
# from sampler import RASampler
# from torch import nn
# from torch.utils.data.dataloader import default_collate
# from torchvision.transforms.functional import InterpolationMode

def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix="", num_iter=5, filename=None):
    batch_size = data_loader.batch_size
        
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

            accuracy = 100 * correct / total

            if i % print_freq == 0:
                print(f'Batch: {i}/{len(data_loader)} Loss: {loss.item()} Accuracy: {correct}/{total} ( {accuracy}% )')
                # print('Batch: {}/{} Loss: {} Accuracy: {}/{} ( {}% )'.format(i, len(data_loader), loss.item(), correct, total, accuracy))

            if i == num_iter:
                return
                
        total_loss /= test_count

        print('Test Accuracy of the model on the 50000 test images: {} %'.format(100 * correct / total))
        print('Average Test loss: {}'.format(total_loss))
