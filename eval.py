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
"""Evaluation without the densecrf with the dice coefficient

    Calculate the Validation score

    Args:
        net: network
        loader: image batch
        device: CUDA/CPU
    Returns:
        Validation score: %
    """
def cal_acc(true_masks,pred_masks):
    pred_masks = torch.argmax(pred_masks,dim=1) 
    # logging.info('correct num: {}'.format((pred_masks == true_masks).sum()))
    return torch.sum(pred_masks == true_masks)/pred_masks.numel()

def eval_net(net, loader, device,force):
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    dice_l = 0
    ignore_labels = [0] # ignore the background only when calculating acuracy
    metric = SegmentationMetric(3,ignore_labels) 
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            frames = batch[0]
            imgs = frames[-1] # B H W C
            mem_list = frames[:-1]
            true_masks = batch[1].unsqueeze(1) # B 1 H W
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
                true_masks_one_hot = F.one_hot(true_masks,3).permute(0,3,1,2).float() # B C H W

                dice_l += dice_loss(mask_pred,true_masks_one_hot,
                                    weight=None,naive_dice=True,ignore_index=0) 
                # calculate metric
                pred_masks = torch.argmax(mask_pred,dim=1) 
                metric.addBatch(pred_masks.cpu(),true_masks.cpu())

            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                dice_l += dice_loss(mask_pred,true_masks,
                                    weight=None,naive_dice=True)
            pbar.update()

    pa = metric.pixelAccuracy()
    mIoU = metric.meanIntersectionOverUnion()
    mIoU1 = metric.meanIntersectionOverUnion1()
    net.train()
    return dice_l/n_val, pa, mIoU, mIoU1, [pred_masks.unsqueeze(1),true_masks.unsqueeze(1),imgs]

def eval_net_flow(net, loader, device,force):
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    dice_l = 0
    ignore_labels = [0] # ignore the background only when calculating acuracy
    metric = SegmentationMetric(3,ignore_labels) 
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            flow = batch[0]
            img = batch[1] # B C H W 
            true_masks = batch[2]# B 1 H W 
            if force:
                frc = batch[3]
                frc = frc.to(device=device, dtype=torch.float32)
            img = img.to(device=device, dtype=torch.float32)
            flow = flow.to(device=device, dtype=torch.float32)
            
            mask_type = torch.float32 if net.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type).squeeze(1)
            with torch.no_grad():
                if force:
                    mask_pred = net(img,flow,frc)
                else:
                    mask_pred = net(img,flow)

            if net.n_classes > 1:
                # calculate dice loss
                true_masks_one_hot = F.one_hot(true_masks,3).permute(0,3,1,2).float() # B C H W

                dice_l += dice_loss(mask_pred,true_masks_one_hot,
                                    weight=None,naive_dice=True,ignore_index=0) 
                # calculate metric
                pred_masks = torch.argmax(mask_pred,dim=1) 
                metric.addBatch(pred_masks.cpu(),true_masks.cpu())

            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                dice_l += dice_loss(mask_pred,true_masks,
                                    weight=None,naive_dice=True)
            pbar.update()

    pa = metric.pixelAccuracy()
    mIoU = metric.meanIntersectionOverUnion()
    mIoU1 = metric.meanIntersectionOverUnion1()
    net.train()
    return dice_l/n_val, pa, mIoU, mIoU1, [pred_masks.unsqueeze(1),true_masks.unsqueeze(1),img]