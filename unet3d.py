import config
from torch.nn import ConvTranspose3d
from torch.nn import Conv3d, MaxPool3d, Sequential, BatchNorm3d, ReLU
from torch.nn import Module
from torch.nn import ModuleList
from torchvision.transforms import CenterCrop
import torchvision.transforms.functional as tf
from torch.nn import functional as F
import torch


class Block(Module):
	'''
	the building unit of our encoder and decoder architecture
	'''
	def __init__(self, inChannels, outChannels):
		super(Block, self).__init__()
		# store the convolution and RELU layers
		self.conv = Sequential(
			Conv3d(inChannels, outChannels, kernel_size=3, stride=1, padding=1, bias=False),
			BatchNorm3d(outChannels),
			ReLU(inplace=True),	#try save some memory with inplace
			Conv3d(outChannels, outChannels, kernel_size=3, stride=1, padding=1, bias=False),
			BatchNorm3d(outChannels),
			ReLU(inplace=True)
		)
		
	def forward(self, x):
		# apply CONV => RELU => CONV block to the inputs and return it
        # x = feature map
		return self.conv(x)



class UNet3D(Module):
	'''
	Following tutorial for 2d UNet: https://www.youtube.com/watch?v=IHq1t7NxS8k
	'''
	def __init__(
			self,
			in_channels = 1,
			out_channels = 1,
			features=[16, 32, 64],	# 64,128,256
	): 
		super(UNet3D, self).__init__()
		self.ups = ModuleList()
		self.downs = ModuleList()
		self.pool = MaxPool3d(kernel_size=2, stride=2)

		# Encoder part
		for feature in features:
			self.downs.append(Block(in_channels, feature))
			in_channels = feature
		
		# Decoder part
		for feature in reversed(features):
			self.ups.append(
				ConvTranspose3d(
					feature*2, feature, kernel_size=2, stride=2
				)
			)
			self.ups.append(Block(feature*2, feature))
		
		# Bottleneck
		self.bottleneck = Block(features[-1], features[-1]*2)

		#Output layer
		self.final_conv = Conv3d(features[0], out_channels, kernel_size=1)
	

	def forward(self, x):
		skip_connections = []

		# Encoder part
		for down in self.downs:
			x = down(x)
			skip_connections.append(x)
			x = self.pool(x)
		
		# Bottleneck part
		x = self.bottleneck(x)

		# Decoder part
		skip_connections = skip_connections[::-1] # reverse order for encoder

	

		for idx in range(0, len(self.ups), 2):
			x = self.ups[idx](x)
			skip_connection = skip_connections[idx//2]

			# dimensions must match for concat
			# this has to be done if input is not divisible by 16
			#if x.shape != skip_connection.shape:
			#	x = tf.resize(x, size=skip_connection.shape[2:])

			concat_skip = torch.cat((skip_connection, x), dim=1)
			x = self.ups[idx+1](concat_skip)
		
		return self.final_conv(x)
		