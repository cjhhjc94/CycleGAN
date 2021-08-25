import torch.optim as optim

def Optimizer(D, G, lr, beta1, beta2):
    d_optimizer = optim.Adam(D.parameters(),lr,[beta1,beta2])
    g_optimizer = optim.Adam(G.parameters(),lr,[beta1,beta2])