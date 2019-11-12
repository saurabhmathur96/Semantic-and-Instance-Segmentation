import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

def Conv2DLayer(input_size, output_size, kernel_size = 3, stride = 1, padding = 1):
  return nn.Sequential(
    nn.Conv2d(input_size, output_size, kernel_size = kernel_size, stride = stride, padding = padding),
    nn.BatchNorm2d(output_size),
    nn.ReLU()
  )

class EncoderBlock(nn.Module):
  def __init__(self, layer_count, input_size, output_size, conv_kwargs, pooling_kwargs):
    super(EncoderBlock, self).__init__()
    self.layer_count = layer_count # # of Conv2DLayers
    self.conv_block = nn.Sequential(
        Conv2DLayer(input_size, output_size, **conv_kwargs),
        *[Conv2DLayer(output_size, output_size, **conv_kwargs) for i in range(layer_count-1)]
    )
    self.pool = nn.MaxPool2d(**pooling_kwargs)
  
  def forward(self, x):
    projected = self.conv_block(x)
    pooled, indices = self.pool(projected)
    return pooled, indices, projected.size()
  
  def initialize_from_layers(self, layers):
    
    for i in range(self.layer_count):
      
      # layer[0] of conv_block[i] is Conv2D layer
      self.conv_block[i][0].weight.data = layers[i].weight.data
      self.conv_block[i][0].bias.data = layers[i].bias.data
      
class Encoder(nn.Module):
  def __init__(self, input_size):
    super(Encoder, self).__init__()
    conv_args = dict(kernel_size = 3, stride = 1, padding = 1)
    pooling_args = dict(kernel_size = 2, stride = 2, padding = 0, return_indices = True)
    self.blocks = nn.ModuleList([
      EncoderBlock(2, input_size, 64, conv_args, pooling_args),
      EncoderBlock(2, 64, 128, conv_args, pooling_args),
      
      EncoderBlock(3, 128, 256, conv_args, pooling_args),
      EncoderBlock(3, 256, 512, conv_args, pooling_args),
      EncoderBlock(3, 512, 512, conv_args, pooling_args)
    ])
    self.block_count = len(self.blocks)
  def forward(self, x):
    pooling_params = []
    for block in self.blocks:
      x, indices, projected_size = block(x)
      pooling_params.append((indices, projected_size))
    return x, pooling_params


class DecoderBlock(nn.Module):
  def __init__(self, layer_count, input_size, output_size, conv_kwargs, pooling_kwargs):
    super(DecoderBlock, self).__init__()
    
    self.unpool = nn.MaxUnpool2d(**pooling_kwargs)
    
    self.layer_count = layer_count # # of Conv2DLayers
    self.conv_block = nn.Sequential(
        Conv2DLayer(input_size, output_size, **conv_kwargs),
        *[Conv2DLayer(output_size, output_size, **conv_kwargs) for i in range(layer_count-1)]
    )
    
  def forward(self, x, indices, output_shape):
    unpooled = self.unpool(output_size=output_shape, indices=indices, input=x)
    return self.conv_block(unpooled)

class Decoder(nn.Module):
  def __init__(self, output_size):
    super(Decoder, self).__init__()
    conv_args = dict(kernel_size = 3, stride = 1, padding = 1)
    pooling_args = dict(kernel_size = 2, stride = 2, padding = 0)
    self.blocks = nn.ModuleList([
      DecoderBlock(3, 512, 512, conv_args, pooling_args),
      DecoderBlock(3, 512, 256, conv_args, pooling_args),
      DecoderBlock(3, 256, 128, conv_args, pooling_args),
      
      DecoderBlock(2, 128, 64, conv_args, pooling_args),
      DecoderBlock(2, 64, output_size, conv_args, pooling_args),
    ])
    # remove last ReLU
    self.blocks[-1].conv_block[-1] = self.blocks[-1].conv_block[-1][:2] 
    self.block_count = len(self.blocks)
  def forward(self, x, pooling_params):
    for params, block in zip(pooling_params, self.blocks):
      x = block(x, *params)
    return x
    

def init_weights(m):
  if type(m) == nn.Conv2d:
    torch.nn.init.xavier_uniform_(m.weight)
    m.bias.data.fill_(0.01)

class SegNet(nn.Module):
  def __init__(self, input_size, class_count):
    super(SegNet, self).__init__()
    
    # Instantiate encoder
    self.encoder = Encoder(input_size)
      
    # Instantiate decoder
    self.decoder = Decoder(class_count)

  def forward(self, x):
    encoded, pooling_params = self.encoder(x)
    decoded = self.decoder(encoded, pooling_params[::-1])
    return decoded
 
    
