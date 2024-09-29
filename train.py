""" train network

Typical usage example:

    see README.md
"""
import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from tmaunet.unet_model import *

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from vessel_loader import *
from datetime import datetime
from mmseg.models.losses.dice_loss import DiceLoss

import random
import numpy as np

dir_root = './data/Multimodal Ultrasound Vascular Segmentation'
dir_checkpoint = 'checkpoints/'

def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    np.random.seed(0 + worker_id)

"""train the network

    Args:
        net: network
        device: CUDA/CPU
        epochs: epochs of trainning
        batch_size: batch
        lr: learning rate
        val_percent: validation image share of dataset
        save_cp: save check point
        img_scale: image scale
        gen_times: generation data round for agumention
    Returns:
        paramters input in terminals
    """
def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              save_cp=True,
              gen_times=0,
              seq_num=1,
              inter=1,
              force=False,
              ):

    if gen_times == 0:
        train =  VesselDataforce(dir_root,augmentations=False,split='train',force=force,path_num=seq_num+1)
        val = VesselDataforce(dir_root,augmentations=False,split='valid',force=force,path_num=seq_num+1)
    else:
        val = VesselDataforce(dir_root,augmentations=False,split='valid',force=force,path_num=seq_num+1)
        train = VesselDataforce(dir_root,augmentations=True,split='train',force=force,path_num=seq_num+1)
        for _ in range(gen_times-1):
            train = torch.utils.data.ConcatDataset([train,VesselDataforce(dir_root,augmentations=True,split='train',force=force,path_num=seq_num+1)])
    n_train = len(train)
    n_val = len(val)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True,worker_init_fn=worker_init_fn)

    sufix = f'_{args.net}_LR_{lr}_BS_{batch_size}_E_{epochs}_G_{gen_times}_S_{seq_num}'
    dir = os.path.join('runs',datetime.now().strftime("%m%d.%H%M")+sufix)
    writer = SummaryWriter(dir)
    global_step = 0


    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {0.5}
        Generation times:{gen_times}
        Net type:        {args.net}
        Seq NUM:         {seq_num}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10,factor=0.7)
    weights = [1, 30.7, 23.1]
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)


    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                frames = batch[0]
                imgs = frames[-1] # B H W C
                mem_list = frames[:-1]
                true_masks = batch[1].unsqueeze(1) # B 1 H W 
                if force:
                    frc = batch[2]
                    frc = frc.to(device=device, dtype=torch.float32)
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mem_list = [x.to(device=device, dtype=torch.float32) for x in mem_list]
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                if force:
                    masks_pred = net(imgs,mem_list,frc)
                else:
                    masks_pred = net(imgs,mem_list)
                loss = criterion(masks_pred, true_masks.squeeze(1))
                epoch_loss += loss.item() 
                writer.add_scalar('Loss/train_ce', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (2 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score,acc,miou,miou1,display = eval_net(net, val_loader, device, force)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        logging.info('Validation dice loss: {}'.format(val_score.item()))
                        logging.info('Validation accuracy: {}'.format(acc))
                        logging.info('Validation miou: {}'.format(miou))
                        writer.add_scalar('Loss/test_dice', val_score, global_step)
                        writer.add_scalar('Acc/test', acc, global_step)
                        writer.add_scalar('MIoU/test', miou, global_step)
                        writer.add_scalar('MIoU/test1', miou1, global_step)

                        writer.add_images('mask/test/pred_img',display[0]/2,global_step) # B C=1/3 H W
                        writer.add_images('mask/test/true_img',display[1]/2,global_step)
                        writer.add_images('image/test',display[2],global_step)
                        writer.add_images('mask/train/pred_img',torch.argmax(masks_pred,dim=1,keepdim=True)/2,global_step)
                        writer.add_images('mask/train/true_img',true_masks/2,global_step)
                        
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)
                    

    if save_cp:
        try:
            os.mkdir(dir_checkpoint)
            logging.info('Created checkpoint directory')
        except OSError:
            pass
        torch.save(net.state_dict(),
                    dir_checkpoint + f'{args.net}_E_{epoch + 1}_G{gen_times}_LR{lr}_BS_{batch_size}_S_{seq_num}.pth')
        logging.info(f'{args.net}_E_{epoch + 1}_G{gen_times}_LR{lr}_BS_{batch_size}_S_{seq_num}.pth saved !')

    writer.close()

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=4,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-g', '--generation', dest='gen', type=int, default=0,
                        help='generate data (0-500) depends on the GPU memory')
    parser.add_argument('-s', '--sequence', dest='seq', type=int, default=2,
                        help='the number of memory frames')
    parser.add_argument('-in', '--interval', dest='inter', type=int, default=1,
                        help='the interval between two frames')   
    parser.add_argument('-fr', '--force', dest='force', type=str, default=False,
                        help='use force or not')
    parser.add_argument('-n', '--net', type=str, default='unet',
                        help='the network') 
    parser.add_argument('--img_size', type=int, default=256, help='input patch size of network input')

    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    seed_everything(seed=0)


    # Change here to adapt to your data
    # n_channels = 3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    n_channels = 3 
    n_classes = 3
    if args.net == 'unet':
        net = UNet(n_channels=3, n_classes=3,bilinear=True)
        require_force = False
    elif args.net == 'fg-unet':
        net = FGUNet(n_channels=3, n_classes=3,sequence_num=args.seq,bilinear=True)
        require_force = True
    elif args.net == 'seq-unet':
        net = SeqUNet(n_channels=3, n_classes=3,sequence_num=args.seq,bilinear=True)
        require_force = False
    

    
    logging.info(f'Network:\n'
                 f'\t{n_channels} input channels\n'
                 f'\t{n_classes} output channels (classes)\n')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  gen_times=args.gen,
                  seq_num=args.seq,
                  inter=args.inter,
                  force=require_force,
                 )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), f'checkpoints/INTERRUPTED_{args.net}.pth')
        logging.info(f'Saved interrupt:INTERRUPTED_{args.net}.pth')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)