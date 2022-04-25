from os import path
import torch

cur_dir = path.dirname(__file__)
root_dir = path.join(cur_dir, 'root')
save_path = path.join(cur_dir, 'result')

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
batch_size = 4
epochs = 3
print_bound = 1000
activation_function = torch.nn.functional.relu