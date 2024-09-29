""" Validation in trainning

Typical usage example:

val_score = eval_net(net, val_loader, device)
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.metric import SegmentationMetric
from utils.dice_loss import dice_coeff
from mmseg.models.losses.dice_loss import dice_loss
import logging
from vessel_loader import *
from torch.utils.data import DataLoader, random_split
from tmaunet.unet_model import *
import cv2

import argparse
frame_num = 1

def cal_acc(true_masks,pred_masks):
    pred_masks = torch.argmax(pred_masks,dim=1) 
    # logging.info('correct num: {}'.format((pred_masks == true_masks).sum()))
    return torch.sum(pred_masks == true_masks)/pred_masks.numel()

def convert_to_rgb_image(array):
    color_map = {
        0: (0, 0, 0),   
        1: (50,147,195), # vein
        2: (178,34,34)    # artery
    }

    h, w, _ = array.shape
    rgb_array = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            pixel_value = int(array[i, j, 0])
            color = color_map.get(pixel_value, (0, 0, 0))
            rgb_array[i, j] = color
    rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2RGB)
    return rgb_array

def eval_net(net, loader, device,force):
    global frame_num
    global size
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)
    dice_l = 0
    ignore_labels = [0] # ignore the background only when calculating acuracy
    metric = SegmentationMetric(3,ignore_labels) 
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            frames = batch[0]
            imgs = frames[-1]
            mem_list = frames[:-1]
            true_masks = batch[1].unsqueeze(1) 
            imgs = imgs.to(device=device, dtype=torch.float32)
            mem_list = [x.to(device=device, dtype=torch.float32) for x in mem_list]
            true_masks = true_masks.to(device=device, dtype=mask_type).squeeze(1)

            if force:
                frc = batch[2]
                frc = frc.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                if force:
                    mask_pred = net(imgs,mem_list,frc)
                else:
                    mask_pred = net(imgs,mem_list)

            if net.n_classes > 1:
                # calculate dice loss
                true_masks_one_hot = F.one_hot(true_masks,3).permute(0,3,1,2).float() 

                dice_l += dice_loss(mask_pred,true_masks_one_hot,
                                    weight=None,naive_dice=True,ignore_index=0) 
                # calculate metric
                pred_masks = torch.argmax(mask_pred,dim=1)  # 1 H W
                metric.addBatch(pred_masks.cpu(),true_masks.cpu())

                # change pred_mask to RGB and resize
                pred_masks_save = pred_masks.permute(1,2,0).cpu().numpy()# H W 1
                pred_masks_save = (pred_masks_save).astype(np.uint8)
                rgb_masks = convert_to_rgb_image(pred_masks_save)
                rgb_masks= cv2.resize(rgb_masks,size)

                # save masks
                cv2.imwrite(os.path.join(out_folder,f"{frame_num}.png"),rgb_masks)
                frame_num = frame_num + 1
                # out.write(pred_masks_save)
                # cv2.imshow('img',rgb_masks)
                # cv2.waitKey(10)

            pbar.update()

    pa = metric.pixelAccuracy()
    mIoU = metric.meanIntersectionOverUnion()
    mIoU1 = metric.meanIntersectionOverUnion1()
    net.train()
    return dice_l/n_val, pa, mIoU, mIoU1, [pred_masks.unsqueeze(1),true_masks.unsqueeze(1),imgs]

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--net', type=str, default='unet',
                        help='the network')
    parser.add_argument('-m','--model',type=str,help='the model parameters')
    return parser.parse_args()

if __name__ == '__main__':
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 4

    # Change paramaters here
    dir_root = './data/Multimodal Ultrasound Vascular Segmentation'
    size = (534,638) # the original size of images
    # out = cv2.VideoWriter('./output/output.avi', fourcc, fps, size)

    args = get_args()
    net_type = args.net
    pth_path = args.model
    if net_type == 'unet':
        # Change checkpoints here
        out_folder = r"./output/test_unet"
        net = UNet(n_channels=3, n_classes=3,bilinear=True)
        require_force = False
    elif net_type == 'fg-unet':
        # Change checkpoints here
        out_folder = r"./output/test_fg-unet"
        net = FGUNet(n_channels=3, n_classes=3,sequence_num=2,bilinear=True)
        require_force = True
    elif net_type == 'seq-unet':
        out_folder = r"./output/test_seq-unet"
        net = SeqUNet(n_channels=3, n_classes=3,sequence_num=2,bilinear=True)
        require_force = False

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    if net_type == 'fg-unet-seq':
        val = VesselDataSequence(dir_root,augmentations=False,split='valid',force=require_force,path_num=3)
    else:
        val = VesselDataforce(dir_root,augmentations=False,split='valid',force=require_force,path_num=3)
    val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.load_state_dict(torch.load(pth_path, map_location=device))
    net.to(device=device)

    logging.info(f'Model loaded from {pth_path}')
    print('loaded')

    dice_l,acc,miou,_,_= eval_net(net,val_loader,device,force=require_force)
    print(f"dice loss: {dice_l},\nacc: {acc},\nmiou: {miou}")