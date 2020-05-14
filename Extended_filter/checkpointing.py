import torch
import os.path as path
from datetime import datetime as dt

def load_checkpoint(G, D, g_optimizer, d_optimizer, date, epoch, name):
    epoch = str(epoch)
    checkpoint = torch.load(path.join('../saved',
                                      '{}_{}_{}'.format(name, date, epoch) ))
    G.load_state_dict(checkpoint['G_state_dict'])
    D.load_state_dict(checkpoint['D_state_dict'])
    g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
    d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
    print('_____Model loaded_____')

def save_checkpoint(G, D, g_optimizer, d_optimizer, epoch, name):
    print('_____Saving_____')
    torch.save({
                'epoch': epoch,
                'G_state_dict': G.state_dict(),
                'D_state_dict': D.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict()
                }, path.join('../saved',
                             '{}_{}_{}'.format(name, dt.today().strftime('%Y-%m-%d'), epoch)))
    print('_____Saved!_____')
