import torch
from thop import clever_format, profile

from nets.unet import UNet

if __name__ == "__main__":
    input_shape = [64, 64]
    num_timesteps = 1000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = UNet(3)
    for i in m.children():
        print(i)
        print("==============================")

    dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    t = torch.randint(0, num_timesteps, (1,), device=device)
    flops, params = profile(m.to(device), (dummy_input, t), verbose=False)

    flops = flops * 2
    flops, params = clever_format([flops, params], "%.3f")
    print("Total GFLOPS: %s" % (flops))
    print("Total params: %s" % (params))
