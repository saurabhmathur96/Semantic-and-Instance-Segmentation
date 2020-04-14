import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
from data import CamVid11
from models import BayesianSegNet, init_weights
from torch.utils.data import DataLoader
from torchvision import models
from tqdm.auto import trange, tqdm



def train_epoch(train_loader, net, criterion, optimizer, device='cuda'):
  net.to(device)
  train_losses = []
  for data, target in tqdm(train_loader):
    data, target = data.permute(0, 3, 1, 2).float().to(device), target.to(device)
    optimizer.zero_grad()
    output = net(data)
    loss = criterion(output, target)
    train_losses.append(loss.item())
    loss.backward()
    optimizer.step()

  return train_losses


if __name__ == '__main__':
  train_loader = DataLoader(CamVid11('CamVid/', split='train'), 
                     batch_size=5, shuffle=True)
  val_loader = DataLoader(CamVid11('CamVid/', split='val'), 
                   batch_size=1, shuffle=False)
  net = BayesianSegNet(3, 12)

  net.apply(init_weights)

  # Initialize encoder weights from VGG16 pre-trained on ImageNet
  vgg16 = models.vgg16(pretrained=True)
  layers = [layer for layer in vgg16.features.children() if isinstance(layer, nn.Conv2d)]

  start = 0
  for i in range(net.encoder.block_count):
    end = start + net.encoder.blocks[i].layer_count
    net.encoder.blocks[i].initialize_from_layers(layers[start:end])
    start = end


  frequencies = {i: [] for i in range(0, 11+1)}

  for _, target in tqdm(train):
    for t in target:
      count = Counter(t.flatten().numpy())
      for key, value in count.items():
        frequencies[key] += [value]
  weights = []
  median = np.median(sum(frequencies.values(), []))
  
  for classid in range(0, 11+1):
    weights.append(median/np.sum(frequencies[classid]))

  criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights))
  optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, weight_decay=0.0005)

  for epoch in trange(100):
    losses = train_epoch(train, net, criterion, optimizer, scheduler)
    print (sum(losses) / len(losses))