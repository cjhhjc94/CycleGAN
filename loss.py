import torch

def real_loss(D_out, smooth=False):
    train_on_gpu = torch.cuda.is_available()

    batch_size=D_out.size(0)
    if smooth:
        labels = torch.ones(batch_size)*0.9
    else:
        labels = torch.ones(batch_size)
        
    if train_on_gpu:
        labels = labels.cuda()
    # criterion = torch.nn.BCEWithLogitsLoss()
    criterion = torch.nn.MSELoss()
    
    loss = criterion(D_out.squeeze(), labels)
    return loss

def fake_loss(D_out):
    train_on_gpu = torch.cuda.is_available()

    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size)
    if train_on_gpu:
        labels = labels.cuda()
    # criterion = torch.nn.BCEWithLogitsLoss()
    criterion = torch.nn.MSELoss()

    loss = criterion(D_out.squeeze(), labels)
    return loss