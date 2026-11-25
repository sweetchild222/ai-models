
import os
import torch
import numpy as np
import torch
import torchvision

from torch.utils.data import DataLoader
from unet import Unet
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

from torchvision.utils import make_grid
from tqdm import tqdm
from diffusion import Diffusion

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data():

    batch_size = 64

    mnist = MNIST(root='./', transform=transforms.ToTensor(), train=True, download=True)
    
    mnist = [((image * 2) - 1).float() for image, _ in mnist]

    return DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)


def create_model():
            
    return Unet().to(device)


def loss_function(noise_pred, noise):

    criterion  = torch.nn.MSELoss()

    return criterion(noise_pred, noise)


def train(model, data, diffusion):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    model.train()
    epochs = 4

    num_timesteps = diffusion.get_num_timesteps()
    
    for epoch in range(epochs):

        losses = []

        for im in tqdm(data):
            
            optimizer.zero_grad()

            im = im.to(device)
            
            noise = torch.randn_like(im).to(device)
            
            t = torch.randint(0, num_timesteps, (im.shape[0],)).to(device)

            noisy_im = diffusion.add_noise(im, noise, t)
            noise_pred = model(noisy_im, t)

            loss = loss_function(noise_pred, noise)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()

        print('epoch: ' + str(epoch + 1) + '/' + str(epochs) + ',\tavg loss: ' + str(np.mean(losses)))


def write_image(xt, path, num_grid_rows):

    ims = torch.clamp(xt, -1., 1.).detach().cpu()
    ims = (ims + 1) / 2
            
    grid = make_grid(ims, nrow=num_grid_rows)
        
    img = torchvision.transforms.ToPILImage()(grid)
    img.save(path)
    img.close()


def generate(model, diffusion):

    model.eval()
    
    with torch.no_grad():

        num_grid_rows = 7
        num_samples =  num_grid_rows**2
        im_channels = 1
        im_size = 28
        num_timesteps = diffusion.get_num_timesteps()

        xt = torch.randn((num_samples, im_channels, im_size, im_size)).to(device)

        for i in reversed(range(num_timesteps)):
            
            noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))

            xt, x0_pred = diffusion.prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
            
            if (i % 10) == 0:

                path = './generated'
                
                if not os.path.exists(path):
                    os.makedirs(path)
                
                write_image(xt, path + '/{}.png'.format(i), num_grid_rows)
                print(path + '/{}.png'.format(i))


num_timesteps = 800
beta_start = 0.0001
beta_end = 0.02

diffusion = Diffusion(num_timesteps=num_timesteps, beta_start=beta_start, beta_end=beta_end)

model = create_model()
data = load_data()
train(model, data, diffusion)

generate(model, diffusion)

