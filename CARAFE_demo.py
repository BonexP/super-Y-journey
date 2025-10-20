import torch
# from mmcv.ops.carafe import CARAFEPack
from carafe import CARAFEPack


x = torch.rand(2, 40, 50, 70)
model = CARAFEPack(channels=40, scale_factor=2)

model = model.cuda()
x = x.cuda()

out = model(x)

print('original shape: ', x.shape)
print('upscaled shape: ', out.shape)

# check pass