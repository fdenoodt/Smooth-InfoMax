import torch

rnd = torch.rand((156198, 15615), device=torch.device("cuda"))
while True:
    rnd = rnd + torch.rand_like(rnd)
