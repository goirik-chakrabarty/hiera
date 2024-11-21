import torch

from hiera import Hiera
from hiera.benchmarking import benchmark
# from MultisessionHiera import MouseHiera


# Hyperparameters

video_size = [36, 64]
batchsize=16

screen_chunk_size = 30
screen_sampling_rate = 30

response_chunk_size = 8
response_sampling_rate = 8

behavior_as_channels = True
replace_nans_with_means = True

dim_head = 64
num_heads = 2
drop_path_rate = 0
mlp_ratio=4

tiny_hiera = Hiera(input_size=(screen_chunk_size, video_size[0], video_size[1]),
                     num_heads=1,
                     embed_dim=96,
                     stages=(2, 1,), # 3 transformer layers 
                     q_pool=1, 
                     in_chans=1,
                     q_stride=(1, 1, 1,),
                     mask_unit_size=(1, 8, 8),
                     patch_kernel=(5, 5, 5),
                     patch_stride=(3, 2, 2),
                     patch_padding=(1, 2, 2),
                     sep_pos_embed=True,
                     drop_path_rate=drop_path_rate,
                     mlp_ratio=4,)

tiny_hiera = tiny_hiera.cuda().to(torch.float32)
example_input = torch.ones(8,1,screen_chunk_size, 36,64).to("cuda", torch.float32)
out = tiny_hiera(example_input, return_intermediates=True)

hiera_output = out[-1][-1]
hiera_output.shape

throughput = benchmark(
    model = tiny_hiera,
    device = 0,
    input_size = example_input.shape[1:],
    batch_size = 16,
    runs = 100,
    throw_out = 0.25,
    use_fp16 = False,
    verbose = True,)