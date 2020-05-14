import torch
from torch import autograd

def train_discriminator(D, d_optimizer, real_data, generated_data, GP):
    with torch.no_grad():
        NG = generated_data.size(0)
        NR = real_data.size(0)

    # Train discriminator
    d_optimizer.zero_grad()
        
    error_d = torch.mean(D(generated_data)) - torch.mean(D(real_data)) + GP
    error_d.backward()
    
    d_optimizer.step()

    return error_d.item()

def train_generator(D, g_optimizer, generated_data):
    with torch.no_grad():
        NG = generated_data.size(0)

    # Train generator
    g_optimizer.zero_grad()

    error_G = -torch.mean(D(generated_data))
    
    error_G.backward()
    g_optimizer.step()

    return error_G.item()

def gradient_penalty(D, generated_labels, real_labels, encoded, seq, Lambda, device, cat):
    alpha = torch.rand(real_labels.shape).to(device)
    interpolates = alpha * real_labels + ((1 - alpha) * generated_labels)
    
    interpolates = cat(interpolates, seq, encoded).requires_grad_(True).to(device)
    disc_interpolates = D(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates,
                              inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True)[0]
    
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * Lambda
   
    return gradient_penalty

