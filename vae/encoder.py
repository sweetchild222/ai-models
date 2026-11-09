import torch.nn as nn


class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)

        
    def forward(self, x):

        h_  = self.LeakyReLU(self.FC_input(x))
        h_  = self.LeakyReLU(self.FC_input2(h_))

        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)

        return mean, log_var
