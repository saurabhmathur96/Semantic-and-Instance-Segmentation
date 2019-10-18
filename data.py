from os import path, listdir
from torch import Tensor
from torchvision.datasets.vision import VisionDataset
from csv import reader
from PIL import Image
import numpy as np
import torch

class CamVid11(VisionDataset):
  def __init__(self, root, split='train', transform=None, target_transform=None, transforms=None):
    super(CamVid11, self).__init__(root, transforms, transform, target_transform)
    self.root = root
    self.split = split

    data_dir = path.join(root, split)
    target_dir = path.join(root, split+'annot')

    images, targets = list(), list()
    mapping_file = path.join(self.root, '%s.txt'%split)
    for data, target in reader(open(mapping_file), delimiter=' '):
      data = data.split('/')[-1]
      images.append(path.join(data_dir, data))

      target = target.split('/')[-1]
      targets.append(path.join(target_dir, target))

    self.images = images
    self.targets = targets

  def __getitem__(self, index):
    image = Image.open(self.images[index]).convert('RGB')
    target = Image.open(self.targets[index])

    return Tensor(np.array(image)).float(), Tensor(np.array(target, int)).long()


  def __len__(self):
    return len(self.images)


