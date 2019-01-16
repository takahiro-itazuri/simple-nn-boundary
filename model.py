import torch
from torch import nn

class Model(nn.Module):
	def __init__(self, nc, nh, nl, activation='relu', use_bn=True, negative_slope=0.1, beta=1.0):
		super(Model, self).__init__()

		if use_bn:
			layers = [
				nn.Linear(2, nh),
				nn.BatchNorm1d(nh),
				get_activation(activation, negative_slope=negative_slope, beta=beta)
			]
		else:
			layers = [
				nn.Linear(2, nh),
				get_activation(activation, negative_slope=negative_slope, beta=beta)
			]

		for i in range(1, nl+1):
			if i == nl:
				layers.extend([
					nn.Linear(nh, nc)
				])
			else:
				if use_bn:
					layers.extend([
						nn.Linear(nh, nh),
						nn.BatchNorm1d(nh),
						get_activation(activation, negative_slope=negative_slope, beta=beta)
					])
				else:
					layers.extend([
						nn.Linear(nh, nh),
						get_activation(activation, negative_slope=negative_slope, beta=beta)
					])
				
		self.model = nn.Sequential(*layers)
	
	def forward(self, x):
		return self.model(x)

def get_activation(activation='relu', negative_slope=0.1, beta=1.0):
	if activation == 'relu':
		return nn.ReLU(inplace=True)
	elif activation == 'leaky_relu':
		return nn.LeakyReLU(negative_slope=negative_slope)
	elif activation == 'tanh':
		return nn.Tanh()
	elif activation == 'sigmoid':
		return nn.Sigmoid()
	elif activation == 'softplus':
		return nn.Softplus(beta=beta)