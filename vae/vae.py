import os
import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder
from model import Model

import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_model(x_dim, hidden_dim, latent_dim):

    encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)
    return Model(Encoder=encoder, Decoder=decoder).to(device)


def load_data():

    batch_size = 100

    dataset = MNIST(root='./dataset', transform=transforms.Compose([transforms.ToTensor()]), train=True, download=True)
    
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4), dataset.data.shape


def loss(x, x_hat, mean, log_var):

    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    
    return reproduction_loss + KLD


def train(data, model):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    epochs = 10
    
    for epoch in range(epochs):

        total_loss = 0
        total_count = 0

        for index, (x, _) in enumerate(data):
            
            x = x.view(x.size(0), -1).to(device)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)

            loss_val = loss(x, x_hat, mean, log_var)

            total_loss += loss_val.item()
            total_count += x.size(0)

            loss_val.backward()
            optimizer.step()
    
        print("epoch: " + str(epoch + 1) + "/" + str(epochs) + "\tloss: " + str(total_loss / total_count))


def generate(model, x_dim, latent_dim):

    model.eval()
    decoder = model.decoder()

    with torch.no_grad():

        image_count = 100

        noise = torch.randn(image_count, latent_dim).to(device)

        generated_images = decoder(noise)

        images = generated_images.view(-1, x_dim, x_dim)

        generated_path = './generated'
        if not os.path.exists(generated_path):
            os.makedirs(generated_path)
        
        for index, image in enumerate(images):
            file_path = generated_path + '/' + str(index) + '.png'
            plt.imsave(file_path, image.cpu().numpy())
            print(file_path)



def main():

    latent_dim = 200
    hidden_dim = 400

    train_data, data_shape = load_data()

    model = create_model(x_dim=data_shape[-1]**2, hidden_dim=hidden_dim, latent_dim=latent_dim)

    train(train_data, model)

    generate(model, x_dim=data_shape[-1], latent_dim=latent_dim)


main()