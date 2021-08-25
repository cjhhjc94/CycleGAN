import os
import pickle as pkl
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.optim as optim
from utils import build_network, scale
from loss import fake_loss, real_loss
from dataloader import get_dataloader
from show_image import imshow
os.environ['KMP_DUPLICATE_LIB_OK']='True'
day_image_dir='day_night_dataset/archive'
night_image_dir='day_night_dataset/archive'
n_epochs=5
print_every=50
d_conv_dim=32
g_conv_dim=32
z_size=100
lr=0.0001
beta1=0.4
beta2=0.99
batch_size = 10
img_size = 128
d_conv_dim = 32
g_conv_dim = 32
z_size = 100
train_on_gpu = torch.cuda.is_available()

daytime_dataloader = get_dataloader('day light road images', day_image_dir, img_size, batch_size)
nighttime_dataloader = get_dataloader('night time road images', night_image_dir, img_size, batch_size)
D, G = build_network(d_conv_dim, g_conv_dim, z_size)
d_optimizer = optim.Adam(D.parameters(),lr,[beta1,beta2])
g_optimizer = optim.Adam(G.parameters(),lr,[beta1,beta2])

#////////////////
# dataiter = iter(daytime_dataloader)
dataiter = iter(nighttime_dataloader)
images, _ = dataiter.next() # _ for no labels
fig = plt.figure(figsize=(10, 4))
plot_size=10
for idx in np.arange(plot_size):
    ax = fig.add_subplot(2, int(plot_size/2), idx+1)
    imshow(images[idx])
plt.show()
#////////////////    print('--------------------------------')


# move models to GPU
if train_on_gpu:
    print('using cuda....')
    D.cuda()
    G.cuda()
else:
    print('using cpu....')
# keep track of loss and generated, "fake" samples
samples = []
losses = []

# Get some fixed data for sampling. These are images that are held
# constant throughout training, and allow us to inspect the model's performance
sample_size=16
fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
fixed_z = torch.from_numpy(fixed_z).float()
# move z to GPU if available
if train_on_gpu:
    fixed_z = fixed_z.cuda()

# epoch training loop
for epoch in range(n_epochs):

    # batch training loop
    for batch_i, (real_images, _) in enumerate(daytime_dataloader):

        batch_size = real_images.size(0)
        real_images = scale(real_images)

        # 1. Train the discriminator on real and fake images
        d_optimizer.zero_grad()
        if train_on_gpu:
            real_images=real_images.cuda()
        d_real=D(real_images)
        d_real_loss=real_loss(d_real,True)
        
        z=np.random.uniform(-1, 1, size=(batch_size, z_size))
        z=torch.from_numpy(z).float()
        if train_on_gpu:
            z = z.cuda()
            
        fake_images=G(z)
        d_fake=D(fake_images)
        d_fake_loss=fake_loss(d_fake)
        d_loss = d_real_loss+d_fake_loss
        d_loss.backward()
        d_optimizer.step()

        # 2. Train the generator with an adversarial loss
        g_optimizer.zero_grad()
        z=np.random.uniform(-1, 1, size=(batch_size, z_size))
        z=torch.from_numpy(z).float()
        if train_on_gpu:
            z = z.cuda()
            
        fake_images=G(z)
        d_fake=D(fake_images)
        
        g_loss = real_loss(d_fake,True)
        g_loss.backward()
        g_optimizer.step()
                    
        # Print some loss stats
        if batch_i % print_every == 0:
            # append discriminator loss and generator loss
            losses.append((d_loss.item(), g_loss.item()))
            # print discriminator and generator loss
            print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    epoch+1, n_epochs, d_loss.item(), g_loss.item()))
    

    ## AFTER EACH EPOCH##    
    # this code assumes your generator is named G, feel free to change the name
    # generate and save sample, fake images
    G.eval() # for generating samples
    samples_z = G(fixed_z)
    samples.append(samples_z)
    G.train() # back to training mode

# Save training generator samples
with open('train_samples.pkl', 'wb') as f:
    pkl.dump(samples, f)
