from torch import nn
import torch.nn.functional as F
from torchvision import models

def Conv2DLayer(input_size, output_size, kernel_size = 3, stride = 1, padding = 1):
  return nn.Sequential(*[
    nn.Conv2d(input_size, output_size, kernel_size = kernel_size, stride = stride, padding = padding),
    nn.BatchNorm2d(output_size),
    nn.ReLU()
  ])

class EncoderBlock(nn.Module):
  def __init__(layer_count, input_size, output_size, conv_kwargs, pooling_kwargs):
    super(EncoderBlock, self).__init__()
    self.layer_count = layer_count # # of Conv2DLayers
    self.conv_block = nn.Sequential(*[
        Conv2DLayer(input_size, output_size, **conv_args),
        *[Conv2DLayer(output_size, output_size, **conv_args) for i in range(layer_count-1)]
    ])
    self.pool = nn.MaxPool2d(*pooling_args)
  
  def forward(self, x):
    projected = self.conv_block(x)
    pooled, indices = self.maxpool(outputs)
    return pooled, indices, projected.size()
  
  def initialize_from_layers(layers):
    for i in range(self.layer_count):
      weight, bias = layers[i]
      
      # layer[0] of conv_block[i] is Conv2D layer
      self.conv_block[i].layer[0].weight.data = weight.data
      self.conv_block[i].layer[0].bias.data = bias.data
      
class Encoder(nn.Module):
  def __init__(self, input_size):
    super(Encoder, self).__init__()
    conv_args = dict(kernel_size = 3, stride = 1, padding = 1)
    pooling_args = dict(kernel_size = 2, stride = 2, padding = 0, return_indices = True)
    self.blocks = [
      EncoderBlock(2, input_size, 64, conv_args, pooling_args),
      EncoderBlock(2, 64, 128, conv_args, pooling_args),
      
      EncoderBlock(3, 128, 256, conv_args, pooling_args),
      EncoderBlock(3, 256, 512, conv_args, pooling_args),
      EncoderBlock(3, 512, 512, conv_args, pooling_args)
    ]
    self.block_count = len(self.blocks)
 def forward(self, x):
  pooling_params = []
  for block in self.blocks:
    x, indices, projected_size = block(x)
    pooling_params.append((indices, projected_size))
  return x, pooling_params


class DecoderBlock(nn.Module):
  def __init__(layer_count, input_size, output_size, conv_kwargs, pooling_kwargs):
    super(DecoderBlock, self).__init__()
    
    self.unpool = nn.MaxUnpool2d(**pooling_kwargs)
    
    self.layer_count = layer_count # # of Conv2DLayers
    self.conv_block = nn.Sequential(*[
        Conv2DLayer(input_size, output_size, **conv_args),
        *[Conv2DLayer(output_size, output_size, **conv_args) for i in range(layer_count-1)]
    ])
    
  
  def forward(self, x, indices, output_shape):
    unpooled = self.unpool(output_size=output_shape, indices=indices, input=x)
    return self.conv_block(unpooled)

class Decoder(nn.Module):
  def __init__(self, output_size):
    super(Decoder, self).__init__()
    conv_args = dict(kernel_size = 3, stride = 1, padding = 1)
    pooling_args = dict(kernel_size = 2, stride = 2, padding = 0)
    self.blocks = [
      DecoderBlock(3, 512, 512, conv_args, pooling_args),
      DecoderBlock(3, 512, 256, conv_args, pooling_args),
      DecoderBlock(3, 256, 128, conv_args, pooling_args),
      
      DecoderBlock(2, 128, 64, conv_args, pooling_args),
      DecoderBlock(2, 64, output_size, conv_args, pooling_args),
    ]
    self.block_count = len(self.blocks)
  
  def forward(self, x, pooling_params):
    for params, block in zip(pooling_params, self.blocks):
      x = block(x, *params)
    return x
    
    
class SegNet(nn.Module):
  def __init__(self):
    super(SegNet, self).__init__()
    
    # Instantiate encoder
    self.encoder = Encoder()
    
    # Initialize encoder weights from VGG16 pre-trained on ImageNet
    vgg16 = models.vgg16(pretrained=True)
    layers = [layer for layer in vgg16.features.children() if isinstance(layer, nn.Conv2d)]
    
    start = 0
    for i in range(self.encoder.block_count):
      end = self.encoder[i].layer_count
      self.encoder[i].initialize_from_layers(layers[start:end])
      start += end
      
    # Instantiate decoder
    self.decoder = Decoder()
  def forward(self, x):
    encoded, pooling_params = self.encoder(x)
    decoded = self.decoder(encoded, pooling_params[::-1])
    return decoded
 
    
