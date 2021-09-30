import time
import os,shutil
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torch.optim.scheduler_
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms


def Train(train):
    import torch.utils.data as data

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch-size', '-N', type=int, default=32, help='batch size')
    train = train
    parser.add_argument(
        '--max-epochs', '-e', type=int, default=1, help='max epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument(
        '--iterations', type=int, default=16, help='unroll iterations')
    parser.add_argument('--checkpoint', type=int, help='unroll iterations')
    args = parser.parse_args()
    
    import dataset
    ## load 32x32 patches from images
    
    train_transform = transforms.Compose([
        transforms.RandomCrop((32, 32)),
        transforms.ToTensor(),
    ])
    
    train_set = dataset.ImageFolder(root=train, transform=train_transform)
    
    train_loader = data.DataLoader(dataset=train_set, batch_size=args.batch_size, 
                                   shuffle=True, num_workers=0)
    
    print("Processing/...")
    print('total images: {}; total batches: {}'.format(
        len(train_set), len(train_loader)))
    
    
    import model
    ## load networks on GPU
    
    encoder = model.EncoderCell()
    binarizer = model.Binarizer()
    decoder = model.DecoderCell()
    
    solver = optim.Adam(
        [
            {
                'params': encoder.parameters()
            },
            {
                'params': binarizer.parameters()
            },
            {
                'params': decoder.parameters()
            },
        ],
        lr=args.lr)
    
    def resume(epoch=None):
        if epoch is None:
            s = 'iter'
            epoch = 0
        else:
            s = 'epoch'
            
        encoder.load_state_dict(
            torch.load('save/encoder_{}_{:08d}.pth'.format(s, epoch)))
        binarizer.load_state_dict(
            torch.load('save/binarizer_{}_{:08d}.pth'.format(s, epoch)))
        decoder.load_state_dict(
            torch.load('save/decoder_{}_{:08d}.pth'.format(s, epoch)))
        
    def save(index, epoch=True):
        if not os.path.exists('save'):
            os.mkdir('save')
        
        if epoch:
            s = 'epoch'
        else:
            s = 'iter'
            
        torch.save(encoder.state_dict(), 'save/encoder_{}_{:08d}.pth'.format(
            s, index))
        torch.save(binarizer.state_dict(),
                   'save/binarizer_{}_{:08d}.pth'.format(s, index))
        torch.save(decoder.state_dict(), 'save/decoder_{}_{:08d}.pth'.format(
            s, index))
        
    # resume()
    scheduler = LS.MultiStepLR(solver, milestones=[3, 10, 20, 50, 100], gamma=0.5)
    last_epoch = 0
    if args.checkpoint:
        resume(args.checkpoint)
        last_epoch = args.checkpoint
        scheduler.last_epoch = last_epoch - 1
    
    for epoch in range(last_epoch + 1, args.max_epochs + 1):
        scheduler.step()
        
        for batch, data in enumerate(train_loader):
            batch_t0 = time.time()
            
            ## init lstm state
            encoder_h_1 = (Variable(torch.zeros(data.size(0), 256, 8, 8)),
                           Variable(torch.zeros(data.size(0), 256, 8, 8)))
            encoder_h_2 = (Variable(torch.zeros(data.size(0), 512, 4, 4)),
                           Variable(torch.zeros(data.size(0), 512, 4, 4)))
            encoder_h_3 = (Variable(torch.zeros(data.size(0), 512, 2, 2)),
                           Variable(torch.zeros(data.size(0), 512, 2, 2)))
            decoder_h_1 = (Variable(torch.zeros(data.size(0), 512, 2, 2)),
                           Variable(torch.zeros(data.size(0), 512, 2, 2)))
            decoder_h_2 = (Variable(torch.zeros(data.size(0), 512, 4, 4)),
                           Variable(torch.zeros(data.size(0), 512, 4, 4)))
            decoder_h_3 = (Variable(torch.zeros(data.size(0), 256, 8, 8)),
                           Variable(torch.zeros(data.size(0), 256, 8, 8)))
            decoder_h_4 = (Variable(torch.zeros(data.size(0), 128, 16, 16)),
                           Variable(torch.zeros(data.size(0), 128, 16, 16)))
            
            patches = Variable(data)
            solver.zero_grad()
            losses = []
            
            res = patches - 0.5
            bp_t0 = time.time()
            
            for _ in range(args.iterations):
                encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(
                    res, encoder_h_1, encoder_h_2, encoder_h_3)
                
                codes = binarizer(encoded)
                
                output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
                    codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)
                
                res = res - output
                losses.append(res.abs().mean())
                
            bp_t1 = time.time()
            
            loss = sum(losses) / args.iterations
            loss.backward()
            
            solver.step()
            batch_t1 = time.time()
            
            print(
                '[TRAIN] Epoch[{}]({}/{}); Loss: {:.6f}; Backpropagation: {:.4f} sec; Batch: {:.4f} sec'.
                format(epoch, batch + 1,
                       len(train_loader), loss.data, bp_t1 - bp_t0, batch_t1 -
                       batch_t0))
            print(('{:.4f} ' * args.iterations +
                   '\n').format(* [l.data for l in losses]))
            
            index = (epoch - 1) * len(train_loader) + batch
            
            ## save checkpoint every 500 training steps
            if index % 500 == 0:
                save(0, False)
        save(epoch)     
    
    '''dir = 'save'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
        torch.optim.scheduler_.save()'''
        
Train('D:\Major-Project\Mine\Code\Dataset')
print("Training Done")
