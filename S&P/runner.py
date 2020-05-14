import torch
import torch.nn as nn

from training import train_discriminator, train_generator, gradient_penalty
from dataloader import get_dataloader
from models import LSTMGenerator, ConvDiscriminator
from checkpointing import load_checkpoint, save_checkpoint
from help_functions import noise, cat_with_seq_with_enc, cat_with_seq_no_enc, cat_no_seq_no_enc, cat_no_seq_with_enc, pad
    
def wgan_gp_run(train_window = 52,
                test_size = 52,
                horizon = 12,
                batch_size = 30,
                epochs = 1,
                d_iterations = 3,
                time_series = True,
                BERT = True,
                load = None,
                cuda_no = 'cuda:0'):

    if torch.cuda.is_available():
        device = torch.device(cuda_no)
        torch.cuda.set_device(device = device)
        FT = torch.cuda.FloatTensor
        use_cuda = True
    else:
        device = torch.device('cpu')
        use_cuda = False
        FT = torch.FloatTensor

    dataloader = get_dataloader(train_window, test_size, batch_size, horizon)
    
    input_size = 1
    if time_series:
        input_size += 1
    if BERT:
        input_size += 768
    
    G = LSTMGenerator(input_size= input_size,
                      hidden_layer_size= 300,
                      num_layers= 2,
                      output_size= 1,
                      horizon = horizon,
                      device = device)

    D = ConvDiscriminator(input_channels = input_size,
                          output_channels = 1,
                          )

    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)

    if use_cuda:
        G.cuda(device = device)
        D.cuda(device = device)

    if not load == None:
        load_epoch = load[1]
        load_checkpoint(G, D,
                        g_optimizer, d_optimizer,
                        date = load[0], epoch = load_epoch,name = load[2])
    else:
        load_epoch = 0
    
    if time_series:
        cat = cat_with_seq_with_enc if BERT else cat_with_seq_no_enc 
    else:
        cat = cat_no_seq_with_enc if BERT else cat_no_seq_no_enc
    
    for epoch in range(load_epoch+1, epochs+load_epoch+1):
        i = 0; j = 0
        for di in range(d_iterations):
            for seq, encoded, labels in dataloader:
                if (i == 1) and (di == 0):
                    print('Currently at Epoch: {}, Discriminator error: {}'.format(epoch, d_error))
                seq = torch.from_numpy(seq).type(FT).unsqueeze(2)
                bz = seq.size(0)
                G.clear_hidden(bz)
                labels = torch.from_numpy(labels).type(FT).unsqueeze(2)
                encoded = torch.from_numpy(encoded).type(FT)
                
                with torch.no_grad(): 
                    generated_labels = pad(G(cat(noise((bz,train_window,1),FT),
                                                 seq,
                                                 encoded)),
                                           train_window,
                                           horizon)

                GP = gradient_penalty(D, generated_labels, labels, encoded, seq, Lambda = 10, device = device, cat = cat)
                d_error = train_discriminator( D,
                                               d_optimizer,
                                               cat(labels, seq, encoded).to(device),
                                               cat(generated_labels, seq, encoded).to(device),GP)
                del labels, generated_labels, encoded, GP
                i += 1

        print('Currently at Epoch: {}, Discriminator error: {}'.format(epoch, d_error))

        for seq, encoded, labels in dataloader:
            if j ==1:
                print('Currently at Epoch: {},  Generator error: {}'.format(epoch, g_error))
            seq = torch.from_numpy(seq).type(FT).unsqueeze(2)
            bz = seq.size(0)
            G.clear_hidden(bz)
            labels = torch.from_numpy(labels).type(FT).unsqueeze(2)
            encoded = torch.from_numpy(encoded).type(FT)

            generated_labels = pad(G(cat(noise((bz,train_window,1), FT),
                                         seq,
                                         encoded) ),
                                  train_window,
                                  horizon)

            g_error = train_generator( D,
                                       g_optimizer,
                                       cat(generated_labels, seq, encoded).to(device))

            del generated_labels, encoded, seq
            j += 1

        print('Currently at Epoch: {},  Generator error: {}'.format(epoch, g_error))

        if (epoch)%5000 == 0: 
            save_checkpoint(G,D,
                            g_optimizer, d_optimizer,
                            epoch,
                            'tw-{}_hz-{}_bs-{}_di-{}_t-{}_B-{}_id-{}_hlr_snp'.format(train_window,
                                                                       horizon,
                                                                       batch_size,
                                                                       d_iterations,
                                                                       time_series,
                                                                       BERT,
                                                                       D.identifier))
