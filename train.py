import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
from data import CamVid11
from models import SegNet, init_weights
from torch.utils.data import DataLoader
from torchvision import models

def train(train_loader, net, criterion, optimizer, scheduler):
  train_losses = []
  for data, target in tqdm(train_loader):
    data, target = data.float(), target
    optimizer.zero_grad()
    output = net(data)
    loss = criterion(output, target)
    train_losses.append(loss.item())
    loss.backward()
    optimizer.step()

  return train_losses


#if __name__ == '__main__':
train = DataLoader(CamVid11('CamVid/', split='train'), 
                   batch_size=5, shuffle=True)
val = DataLoader(CamVid11('CamVid/', split='val'), 
                 batch_size=1, shuffle=True)
net = SegNet(3, 12)

net.apply(init_weights)

# Initialize encoder weights from VGG16 pre-trained on ImageNet
vgg16 = models.vgg16(pretrained=True)
layers = [layer for layer in vgg16.features.children() if isinstance(layer, nn.Conv2d)]

start = 0
for i in range(net.encoder.block_count):
  end = start + net.encoder.blocks[i].layer_count
  net.encoder.blocks[i].initialize_from_layers(layers[start:end])
  start = end

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-6, weight_decay=0.001)
scheduler = ExponentialLR(optimizer, gamma=0.9999)

