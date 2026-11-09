import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model(nn.Module):

    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()

        self.Encoder = Encoder
        self.Decoder = Decoder


    def reparameterization(self, mean, var):
        
        epsilon = torch.randn_like(var).to(device)
        z = mean + var * epsilon
        return z

        
    def forward(self, x):

        mean, log_var = self.Encoder(x)        
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.Decoder(z)
        
        return x_hat, mean, log_var


    def decoder(self):

        return self.Decoder

    def encoder(self):
        
        return self.Encoder
    
    