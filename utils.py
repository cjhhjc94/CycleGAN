import torch
import numpy as np
from model import Discriminator, Generator


def weights_init_normal(m):
    classname = m.__class__.__name__

    if 'Conv' in classname:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'Linear' in classname:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

def build_network(d_conv_dim, g_conv_dim, z_size):
    # define discriminator and generator
    D = Discriminator(d_conv_dim)
    G = Generator(z_size=z_size, conv_dim=g_conv_dim)

    # initialize model weights
    D.apply(weights_init_normal)
    G.apply(weights_init_normal)

    # print(D)
    # print()
    # print(G)
    
    return D, G
    
def scale(x, feature_range=(-1, 1)):
    x=x*2-1
    return x
    