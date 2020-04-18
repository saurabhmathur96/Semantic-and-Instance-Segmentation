from os import path, listdir
from torch import Tensor
from torchvision.datasets.vision import VisionDataset
from csv import reader
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms


import random

from PIL import Image, ImageOps

def iou(pred, target, classes):
  ious = []
  pred = pred.view(-1)
  target = target.view(-1)

  # Ignore IoU for background class ("0")
  for c in classes:  # This goes from 1:n_classes-1 -> class "0" is ignored
    pred_inds = pred == c
    target_inds = target == c
    intersection = (pred_inds[target_inds]).long().sum().data.cpu()[0]  # Cast to long to prevent overflows
    union = pred_inds.long().sum().data.cpu()[0] + target_inds.long().sum().data.cpu()[0] - intersection
    if union == 0:
      ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
    else:
      ious.append(float(intersection) / float(max(union, 1)))
  return np.array(ious)

class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


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
    image = Tensor(np.array(image)).float()
    image = image.permute(2, 0, 1) / 255.
    if self.transform is not None:
      image = self.transform(image)
    target = Image.open(self.targets[index])
    target = Tensor(np.array(target, int)).long()
    return image, target


  def __len__(self):
    return len(self.images)

