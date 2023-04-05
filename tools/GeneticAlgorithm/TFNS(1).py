

import numpy as np
import torch
import torch.utils.data as Data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from VGG16_Model import vgg
from TDE import FDE

batch_size = 1
test_dataset = dsets.CIFAR10(root='/home/amax/文档/LICHAO_code/lc_code_CIFAR10_Non_target/CIFAR_data',
                                download=False,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                            ]),
                                train=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = vgg().to(device)
model.load_state_dict(torch.load(r'C:\Users\LC\Desktop\AGSM-DE\AGSM-DE-PythonCode\vgg16_params.pkl',map_location=torch.device('cuda')))

count = 0
total_count = 0
net_correct = 0

model.eval()

for images, labels in test_loader:
      images = images.to(device)
      labels = labels.to(device)
      outputs = model(images)
      _, pre = torch.max(outputs.data, 1)
      total_count += 1
      if pre == labels:
           net_correct += 1
           images = FDE(images, labels)
           images = images.to(device)
           outputs = model(images)
           _, pre = torch.max(outputs.data, 1)
           if pre == labels:
                count += 1
      acctak_count = net_correct - count
      print(total_count, net_correct, count, acctak_count)
      if net_correct > 0:
          print('Accuracy of attack: %f %%' % (100 * float(acctak_count) / net_correct))
      if net_correct ==500:
         break