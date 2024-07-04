import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
import torch.nn.utils.prune as prune
import torch.nn.functional as F


def create_dataloader():
	class RandomDataset(Dataset):
		def __init__(self, num_samples, num_channels, height, width):
			self.data = torch.randn(num_samples, num_channels, height, width)
			self.labels = torch.randint(0, 10, (num_samples,))  # 假设有10个类别

		def __len__(self):
			return len(self.data)

		def __getitem__(self, idx):
			return self.data[idx], self.labels[idx]

	# 参数设置
	num_samples = 1000  # 样本数量
	num_channels = 1  # 通道数量，LeNet模型接受单通道输入
	height = 32  # 图像高度
	width = 32  # 图像宽度
	batch_size = 32  # 批量大小

	# 创建数据集和数据加载器
	random_dataset = RandomDataset(num_samples, num_channels, height, width)
	dataloader = DataLoader(random_dataset, batch_size=batch_size, shuffle=True)

	return dataloader


def compute_channel_importance(model: torch.nn.Module, dataloader: DataLoader):
	"""
	This method shows how to compute:
		- neuron importance scores based on loss according to http://arxiv.org/abs/1905.10650
	"""
	# collect weights
	conv_weight = []
	channel_importance = {}
	for name, w in model.named_parameters():
		# todo 按需要修改的 Module name
		if 'conv' in name:
			conv_weight.append(w)
			channel_importance[name.replace('.weight', '')] = torch.zeros(w.shape[0]).cuda()

	criterion = torch.nn.CrossEntropyLoss()

	# switch to train mode
	model.train()

	# for i in dataloader:
	# 	print(i)

	# Input the training set to obtain the weight gradient and rearrange the channel importance.
	for images, target in dataloader:
		# images = images.cuda()
		# target = target.cuda()
		images = images.to(device)
		target = target.to(device)

		output = model(images)
		loss = criterion(output, target)

		loss.backward()

		# Calculate channel importance
		# The importance of the output channel is defined as: the sum of all weights * weight gradient in this channel
		for w, current_importance in zip(conv_weight, channel_importance.values()):
			current_importance += ((w * w.grad).sum(dim=(1, 2, 3))).abs().detach()

	# todo: remove global

	# Rank channel importance in convolutional layers based on Taylor importance score
	reordered_channels = {}
	for k, v in channel_importance.items():
		reordered_channels[k] = torch.sort(v, descending=True)[-1]
	return reordered_channels


def channel_prune(module):
	pass


class TylarPruningMethod(prune.BasePruningMethod):
	PRUNING_TYPE = 'Tylar'

	def compute_mask(self, t, default_mask):
		mask = default_mask.clone()
		tmp = compute_channel_importance()


# ???
# mask.view()[,:,:,:]=0


# TODO: 实现应用方法


def pruning(module, name: ["weight", "bias"]):
	TylarPruningMethod.apply(module, name)


if __name__ == '__main__':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	class LeNet(nn.Module):
		def __init__(self):
			super(LeNet, self).__init__()
			# 1 input image channel, 6 output channels, 5x5 square conv kernel
			self.conv1 = nn.Conv2d(1, 6, 5)
			self.conv2 = nn.Conv2d(6, 16, 5)
			self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
			self.fc2 = nn.Linear(120, 84)
			self.fc3 = nn.Linear(84, 10)

		def forward(self, x):
			x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
			x = F.max_pool2d(F.relu(self.conv2(x)), 2)
			x = x.view(-1, int(x.nelement() / x.shape[0]))
			x = F.relu(self.fc1(x))
			x = F.relu(self.fc2(x))
			x = self.fc3(x)
			return x


	model = LeNet().to(device=device)
	dataloader = create_dataloader()
	# weight,bias = model.conv1.named_parameters()
	# print(f'weight	:{weight}\n')
	compute_channel_importance(model, dataloader)
	pruning(model.conv1, name='bias')

	# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
	# data = torch.randn(1, 3, 224, 224, requires_grad=True)
	# torch.onnx.export(model, data, '../resnet.onnx', opset_version=12, input_names="torch.randn")
	# torch.onnx.export(model,  # 模型
	# 				  data,  # 输入张量
	# 				  "resnet18.onnx",  # 输出的ONNX文件路径
	# 				  export_params=True,  # 是否导出参数到文件中
	# 				  opset_version=12,  # 指定ONNX的操作集版本
	# 				  do_constant_folding=True,  # 是否执行常量折叠优化
	# 				  input_names=['input'],  # 输入张量的名称
	# 				  output_names=['output'],  # 输出张量的名称
	# 				  dynamic_axes={'input': {0: 'batch_size'},  # 可变长度轴
	# 								'output': {0: 'batch_size'}})
	if False:
		for name, w in model.named_parameters():
			print(f'name: {name}\n')
		# module = model.conv1
		# print(list(module.named_parameters()))
		# print(list(module.named_buffers()))
		print("=" * 20)

# prune.random_unstructured(module, name="weight", amount=0.3)
# print(module.weight)

# print(list(module.named_parameters()))
# print("=" * 20)

# for data in dataloader:
# 	model.train()
# 	pred = model(data)
