import os
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# def get_dataloader(batch_size, image_size, data_dir):
#     transform = transforms.Compose([transforms.Resize(image_size),
#                                     transforms.ToTensor()])
#     image_path = data_dir
#     # define datasets using ImageFolder
#     train_dataset = datasets.ImageFolder(image_path, transform)
#     # create and return DataLoaders
#     train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

#     return train_loader

def get_dataloader(image_type, image_dir, image_size, batch_size):
    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.ToTensor()])
    image_path = 'D:/' + image_dir
    train_path = os.path.join(image_path, image_type)    # define datasets using ImageFolder
    train_dataset = datasets.ImageFolder(train_path, transform)
    # create and return DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader

