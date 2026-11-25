import torch


class Diffusion:

    def __init__(self, num_timesteps, beta_start, beta_end):

        self.num_timesteps = num_timesteps

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1. - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)


    def get_num_timesteps(self):

        return self.num_timesteps

    
    def add_noise(self, x, noise, t):

        batch_size = x.shape[0]
        
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(x.device)[t].reshape(batch_size)
        sqrt_minus_alpha_cum_prod = self.sqrt_minus_alpha_cum_prod.to(x.device)[t].reshape(batch_size)
        
        for _ in range(len(x.shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
        for _ in range(len(x.shape) - 1):
            sqrt_minus_alpha_cum_prod = sqrt_minus_alpha_cum_prod.unsqueeze(-1)
        
        return (sqrt_alpha_cum_prod.to(x.device) * x + sqrt_minus_alpha_cum_prod.to(x.device) * noise)


    def prev_timestep(self, xt, noise_pred, t):

        x0 = ((xt - (self.sqrt_minus_alpha_cum_prod.to(xt.device)[t] * noise_pred)) / torch.sqrt(self.alpha_cum_prod.to(xt.device)[t]))
        x0 = torch.clamp(x0, -1., 1.)
        
        mean = xt - ((self.betas.to(xt.device)[t]) * noise_pred) / (self.sqrt_minus_alpha_cum_prod.to(xt.device)[t])
        mean = mean / torch.sqrt(self.alphas.to(xt.device)[t])
        
        if t == 0:
            return mean, x0
        else:
            variance = (1.0 - self.alpha_cum_prod.to(xt.device)[t - 1]) / (1.0 - self.alpha_cum_prod.to(xt.device)[t])
            variance = variance * self.betas.to(xt.device)[t]
            sigma = variance ** 0.5
            z = torch.randn(xt.shape).to(xt.device)
            
            return mean + sigma * z, x0
