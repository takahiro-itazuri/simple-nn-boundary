import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from model import Model

class TensorDatasetWithLabel(Dataset):
	def __init__(self, tensors, labels):
		self.tensors = tensors
		self.labels = torch.LongTensor(labels)

	def __getitem__(self, index):
		return (self.tensors[index], self.labels[index])

	def __len__(self):
		return self.tensors.size(0)

def generate_dataset(means, std, nsamples):
	data = []
	labels = []
	for l, mean in enumerate(means):
		data.append( torch.FloatTensor(mean) + std * torch.randn(nsamples, 2))
		labels.extend([l] * nsamples)
	tensors = torch.cat(data)
	dataset = TensorDatasetWithLabel(tensors, labels)
	return tensors, labels, dataset

if __name__ == '__main__':
	device = 'cuda'
	nsamples = 10000
	std = 0.25
	means = [[1.0, 0.0], [-1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
	tensors, labels, dataset = generate_dataset(means, std, nsamples)

	nc = len(means)
	nl = 10
	nh = 10
	lr = 0.01
	momentum = 0.9
	model = Model(nc, nh, nl, activation='softplus', use_bn=True, beta=100.).to(device)
	optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
	criterion = nn.CrossEntropyLoss()

	nepochs = 1000
	checkpoint = 10
	batch_size = 10000
	loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
	for epoch in range(1, nepochs+1):
		total = 0
		correct = 0
		for itr, (x, t) in enumerate(loader):
			optimizer.zero_grad()
			x, t = x.to(device), t.to(device)
			y = model(x)
			loss = criterion(y, t)
			loss.backward()
			optimizer.step()

			_, pred = y.max(1)
			total += x.size(0)
			correct += pred.eq(t).sum().item()

		if epoch % checkpoint == 0:
			print('[{:d}] acc: {:.4f}, loss: {:.2e}'.format(epoch, float(correct)/float(total), loss.item()/float(total)))

	model.eval()
	grid_size = 10000
	x = np.linspace(-2.0, 2.0, grid_size)
	y = np.linspace(-2.0, 2.0, grid_size)
	xx, yy = np.meshgrid(x, y)
	np_grid = np.c_[xx.ravel(), yy.ravel()]
	torch_grid = torch.from_numpy(np_grid.astype(np.float32)).to(device)

	results = []
	for i in range(grid_size**2 // batch_size):
		o = torch.argmax(model(torch_grid[i*batch_size:(i+1)*batch_size]), dim=1).cpu().detach()
		results.append(o)
	results = torch.cat(results).numpy()
	results = np.reshape(results, (grid_size, grid_size))
	cmap = plt.get_cmap('tab10')
	r = np.zeros((grid_size, grid_size, 4))
	for i in range(grid_size):
		for j in range(grid_size):
			c = list(cmap(results[i][j]))
			c[3] = 0.5
			r[i][j] = c
	plt.figure(figsize=(10,10))
	plt.imshow(r, origin='lower')

	numpys = tensors.numpy()
	numpys = numpys * (grid_size / 4) + (grid_size / 2)
	for i in range(len(means)):
		plt.scatter(numpys[i*nsamples:(i+1)*nsamples, 0], numpys[i*nsamples:(i+1)*nsamples, 1], s=2, c=cmap(i))
	plt.xticks(color="None")
	plt.yticks(color="None")
	plt.xlim(0, grid_size)
	plt.ylim(0, grid_size)
	plt.gca().spines['top'].set_visible(False)
	plt.gca().spines['bottom'].set_visible(False)
	plt.gca().spines['right'].set_visible(False)
	plt.gca().spines['left'].set_visible(False)
	plt.tick_params(length=0)
	plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
	plt.show()
