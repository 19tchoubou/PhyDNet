import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.load import load_dataset
from models.models import ConvLSTM, PhyCell, EncoderRNN
from constrain_moments import K2M
from pathlib import Path


device = torch.device("cuda")
print('Running on ', torch.cuda.get_device_name())
loader = load_dataset(path = Path.cwd()/'data'/'reprocessed_seg_frames', batch_size = 8, T = 10)

def train_on_batch(input_tensor, target_tensor, encoder, encoder_optimizer, criterion,teacher_forcing_ratio):                
    encoder_optimizer.zero_grad()
    # input_tensor : torch.Size([batch_size, input_length, 1, 64, 64])
    input_length  = input_tensor.size(1)
    target_length = target_tensor.size(1)
    loss = 0
    for ei in range(input_length-1): 
        encoder_output, encoder_hidden, output_image,_,_ = encoder(input_tensor[:,ei,:,:,:], (ei==0) )
        loss += criterion(output_image,input_tensor[:,ei+1,:,:,:])

    decoder_input = input_tensor[:,-1,:,:,:] # first decoder input = last image of input sequence
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False 
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, output_image,_,_ = encoder(decoder_input)
            target = target_tensor[:,di,:,:,:]

            loss += criterion(output_image, target)
            decoder_input = target 
         
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, output_image,_,_ = encoder(decoder_input)
            decoder_input = output_image
            target = target_tensor[:,di,:,:,:]
            loss += criterion(output_image, target)
 
    # Moment Regularisation  encoder.phycell.cell_list[0].F.conv1.weight # size (nb_filters,in_channels,7,7)
    k2m = K2M([7,7]).to(device)
    for b in range(0,encoder.phycell.cell_list[0].input_dim):
        filters = encoder.phycell.cell_list[0].F.conv1.weight[:,b,:,:] # (nb_filters,7,7)
        
        m = k2m(filters.double()) 
        m  = m.float()   
        loss += criterion(m, constraints) # constrains is a precomputed matrix   
    loss.backward()
    encoder_optimizer.step()
    return loss.item() / target_length

def trainIters(encoder, train_loader, n_epochs):
    start = time.time()
    train_losses = []

    encoder_optimizer = torch.optim.Adam(encoder.parameters(),lr=0.0001)
    criterion = nn.MSELoss()
    
    for epoch in range(n_epochs):
        t0 = time.time()
        loss_epoch = 0
        teacher_forcing_ratio = max(0 , 1 - epoch * 0.01)
        
        for i, out in enumerate(train_loader, 0):
            #out -> torch.size([8, 10, 1, 128, 128])
            input_tensor = out[:, :5, :, :, :].to(device)
            target_tensor = out[:, 5:, :, :, :].to(device)
            loss = train_on_batch(input_tensor, target_tensor, encoder, encoder_optimizer, criterion, teacher_forcing_ratio)                                   
            loss_epoch += loss
        train_losses.append(loss_epoch)        
        print('epoch ',epoch,  ' loss ',loss_epoch , ' epoch time ',time.time()-t0)
                           
    return train_losses

k2m = K2M([7, 7]).to(device)
constraints = torch.zeros((49,7,7)).to(device)
ind = 0
for i in range(0,7):
    for j in range(0,7):
        constraints[ind,i,j] = 1
        ind +=1    

phycell =  PhyCell(input_shape=(32,32), input_dim=64, F_hidden_dims=[49], n_layers=1, kernel_size=(7,7), device=device) 
convlstm =  ConvLSTM(input_shape=(32,32), input_dim=64, hidden_dims=[128,128,64], n_layers=3, kernel_size=(3,3), device=device)   
encoder = EncoderRNN(phycell, convlstm, device)

if __name__ == '__main__':
    trainIters(encoder, loader, 100)
