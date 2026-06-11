import torch
print("cuda available:", torch.cuda.is_available(), flush=True)
print("device:", torch.cuda.get_device_name(0), flush=True)
print("tensor math:", (torch.randn(3).cuda() * 2), flush=True)
print("arch list:", torch.cuda.get_arch_list(), flush=True)