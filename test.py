import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


# Optionally use the context manager to ensure one of the fused kernels is run
query = torch.rand(8, 1, 90, 64, 96, dtype=torch.float16, device="cuda")
key = torch.rand(8, 1, 90, 64, 96, dtype=torch.float16, device="cuda")
value = torch.rand(8, 1, 90, 64, 96, dtype=torch.float16, device="cuda")
with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
    F.scaled_dot_product_attention(query,key,value)