import torch
import torch.nn as nn
import torch.nn.functional as F

def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                           kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    layers = []
    conv_layer = nn.ConvTranspose2d(in_channels, out_channels, 
                           kernel_size, stride, padding, bias=False)
    
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)

class Discriminator(nn.Module):
    
    def __init__(self, conv_dim):
        super(Discriminator, self).__init__()
        self.conv_dim=conv_dim        #32x32
        self.conv1=conv(3,conv_dim,4,batch_norm=False)#16xx16
        self.conv2=conv(conv_dim,conv_dim*2,4)#8x8
        self.conv3=conv(conv_dim*2,conv_dim*4,4)#4x4
        self.fc=nn.Linear(conv_dim*4*4*4,1)
        self.dropout=nn.Dropout(0.4)
        
    def forward(self, x):
        x=F.leaky_relu(self.conv1(x),0.1)
        x=self.dropout(x)
        x=F.leaky_relu(self.conv2(x),0.1)
        x=self.dropout(x)
        x=F.leaky_relu(self.conv3(x),0.1)
        x=x.view(-1,self.conv_dim*4*4*4)
        x=self.fc(x)
        return x

class Generator(nn.Module):
    def __init__(self, z_size, conv_dim):
        super(Generator, self).__init__()
        self.conv_dim=conv_dim
        self.z_size=z_size
        self.fc=nn.Linear(z_size,conv_dim*32*4*4) #4x4
        self.deconv1=deconv(conv_dim*32,conv_dim*16,4) #8x8
        self.deconv2=deconv(conv_dim*16,conv_dim*8,4) #16x16
        self.deconv3=deconv(conv_dim*8,conv_dim*4,4) #32x32
        self.deconv4=deconv(conv_dim*4,3,3,stride=1,batch_norm=False) #32x32
        self.dropout=nn.Dropout(0.4)        

    def forward(self, x):
        x=self.fc(x)
        x=x.view(x.size()[0],self.conv_dim*32,4,4)
        x=F.leaky_relu(self.deconv1(x),0.1)
        x=self.dropout(x)
        x=F.leaky_relu(self.deconv2(x),0.1)
        x=self.dropout(x)
        x=F.leaky_relu(self.deconv3(x),0.1)
        x=torch.tanh(self.deconv4(x))
        
        return x