import sys

import torch
import torch.nn as nn

sys.path.insert(0, '..')
import torchattacks

sys.path.insert(0, '..')
import robustbench
from robustbench.data import load_cifar10
from robustbench.utils import load_model, clean_accuracy

images, labels = load_cifar10(n_examples=5)
print('[Data loaded]')
print(images.shape)
device = "cuda"
model = load_model('Standard', norm='Linf').to(device)
acc = clean_accuracy(model, images.to(device), labels.to(device))
print('[Model loaded]')
print('Acc: %2.2f %%'%(acc*100))
