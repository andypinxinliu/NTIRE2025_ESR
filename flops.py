from utils.model_summary import get_model_flops, get_model_activation
from models.team00_EFDN import EFDN
from fvcore.nn import FlopCountAnalysis
import torch

device = 'cuda'
model = EFDN().to('cuda')

input_dim = (3, 256, 256)  # set the input dimension
activations, num_conv = get_model_activation(model, input_dim)
activations = activations / 10 ** 6
print("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
print("{:>16s} : {:<d}".format("#Conv2d", num_conv))

# The FLOPs calculation in previous NTIRE_ESR Challenge
# flops = get_model_flops(model, input_dim, False)
# flops = flops / 10 ** 9
# print("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

# fvcore is used in NTIRE2025_ESR for FLOPs calculation
input_fake = torch.rand(1, 3, 256, 256).to(device)
flops = FlopCountAnalysis(model, input_fake).total()
flops = flops/10**9
print("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
num_parameters = num_parameters / 10 ** 6
print("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))