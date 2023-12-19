import torch

current_file = '/'.join(__file__.split('/')[:-1])

top_10k = torch.load(f'{current_file}/top_10k_indices.pt')