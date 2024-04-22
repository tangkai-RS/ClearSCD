import itertools

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from dataset.dataset import HiUCDDataset, LoveDADataset, LoveDADataset_for_SCD, NanjingDataset
from loss.losses import BCDLoss, AdditionalBackgroundSupervision
from utils.callbacks import AverageMeter
from utils.evaluator import BCDEvaluator, SEGEvaluator
from utils.helper import get_lr, seed_torch, get_model
from utils.parser import get_parser_with_args_from_json
from utils.saver import Saver
from utils.logger import Logger as Log


def split_sample(sample, seg_pretrain=False):
    if not seg_pretrain:
        img_A = sample['img_A'].cuda(non_blocking=True)
        img_B = sample['img_B'].cuda(non_blocking=True)
        label_BCD = sample['label_BCD'].cuda(non_blocking=True)
        label_SGA = sample['label_SGA'].cuda(non_blocking=True)
        label_SGB = sample['label_SGB'].cuda(non_blocking=True)   
        return img_A, img_B, label_BCD, label_SGA.long(), label_SGB.long()
    else:
        imgs = sample['img_A'].cuda(non_blocking=True)
        labels = sample['label_SGA'].cuda(non_blocking=True)
        batch_size = int(imgs.shape[0] / 2)
        img_A = imgs[0:batch_size, :]
        img_B = imgs[batch_size::, :]    
        label_SGA = labels[0:batch_size, :]
        label_SGB = labels[batch_size::, :] 
        return img_A, img_B, None, label_SGA.long(), label_SGB.long()


def get_dataset(args):
    if args.dataset == 'HiUCD':
        train_dataset = HiUCDDataset(args, split='train')
        val_dataset = HiUCDDataset(args, split='val')
    elif args.dataset =='LoveDA':
        train_dataset = LoveDADataset(args, split='train')
        val_dataset = LoveDADataset(args, split='val')
    elif args.dataset == 'LoveDAforSCD':
        train_dataset = LoveDADataset_for_SCD(args, split='train')
        val_dataset = LoveDADataset_for_SCD(args, split='val')        
    elif args.dataset == 'Nanjing':
        train_dataset = NanjingDataset(args, split='train')
        val_dataset = NanjingDataset(args, split='val')            
    return train_dataset, val_dataset


def main(args):  
    seed_torch()
     
    model = get_model(args).cuda()
    train_dataset, val_dataset = get_dataset(args)
    
    drop_last = True
    if args.dataset == 'Nanjing':
        drop_last = False
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              persistent_workers=True, pin_memory=True, num_workers=args.num_workers, drop_last=drop_last)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, 
                            persistent_workers=True, pin_memory=True, num_workers=args.val_num_workers, drop_last=drop_last)
    
    loss_bcd = BCDLoss()
    loss_seg = torch.nn.CrossEntropyLoss(ignore_index=args.num_segclass) 
    loss_abs = AdditionalBackgroundSupervision(ignore_index=args.num_segclass)
    optimizer = torch.optim.Adam(itertools.chain(model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
      
    saver = Saver(args)    
    evaluator_bcd = BCDEvaluator()
    evaluator_seg_A = SEGEvaluator(args.num_segclass)
    evaluator_seg_B = SEGEvaluator(args.num_segclass)
    evaluator_seg_total = SEGEvaluator(args.num_segclass)
    metric_best = -1
    metric_best_dict = {}
    start_epoch = 1
    Log.init(logfile_level="info", log_file=saver.experiment_dir + '/log.log')

    if isinstance(args.resume, str):
        checkpoint = torch.load(args.resume)
        checkpoint['epoch'] = 1   
        model.load_state_dict(checkpoint['state_dict']) 
        Log.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        del checkpoint      
    epoch_best = start_epoch   
    
    for epoch in tqdm(range(start_epoch, args.epochs + 1), desc=args.congfig_name):     
        if epoch > args.warmup_epoch:
            warmup = False
        else:
            warmup = True
                    
        # traning 
        losses_seg_A = AverageMeter()
        losses_seg_B = AverageMeter()
        losses_bcd = AverageMeter()
        losses_bscc = AverageMeter()
        losses_cont = AverageMeter()
        losses_total = AverageMeter()
        
        model.train()
        for batch_idx, sample in enumerate(train_loader):
            # wheter seg pretraining
            img_A, img_B, label_BCD, label_SGA, label_SGB = split_sample(sample, seg_pretrain=args.seg_pretrain)
            imgs = torch.cat([img_A, img_B], dim=1) 
            
            # whether contrastive learning
            if args.with_sacl:
                outputs = model(imgs, label_SGA, label_SGB, label_BCD, test=False, warmup=warmup)
            elif args.with_bscc:
                outputs = model(imgs, label_BCD=label_BCD, test=False)
            else:
                outputs = model(imgs)
            
            # whether only segmentation
            if not args.only_seg:
                loss_cd = loss_bcd(outputs['BCD'], label_BCD)
                
            # whether seg pretraining
            elif args.only_seg and args.seg_pretrain:
                # whether using additional background supervision
                if args.with_abs:
                    loss_cd = loss_abs(torch.cat([outputs['seg_A'], outputs['seg_B']], dim=0), torch.cat([label_SGA, label_SGB], dim=0))
                else:
                    loss_cd = torch.tensor(0)
            else:
                loss_cd = torch.tensor(0)
                
            # whether only binary change detection or seg_pretrain
            if not args.only_bcd and (not args.seg_pretrain):
                loss_seg_A = loss_seg(outputs['seg_A'], label_SGA)
                loss_seg_B = loss_seg(outputs['seg_B'], label_SGB)
            elif not args.only_bcd and args.seg_pretrain:
                loss_seg_A = loss_seg(torch.cat([outputs['seg_A'], outputs['seg_B']], dim=0), torch.cat([label_SGA, label_SGB], dim=0))
                loss_seg_B = torch.tensor(0)
            elif args.only_bcd:
                loss_seg_A = torch.tensor(0)
                loss_seg_B = torch.tensor(0)        
                        
             # only global: must not warmup; only local: anyway
            flag = (not warmup) or args.local_contrast
            if args.with_sacl and flag:
                loss_contrast = outputs['contrast_loss']
                losses_cont.update(loss_contrast.item())
            else:
                loss_contrast = torch.tensor(0)
    
            # bscc loss
            if args.with_bscc:
                loss_bscc = outputs['bscc_loss']   
            else:
                loss_bscc = torch.tensor(0)
            # total loss   
            loss = loss_cd + loss_seg_A + loss_seg_B + loss_contrast + loss_bscc
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses_bcd.update(loss_cd.item())    
            losses_bscc.update(loss_bscc.item())        
            losses_seg_A.update(loss_seg_A.item())
            losses_seg_B.update(loss_seg_B.item())
            losses_total.update(loss.item())
                
            if batch_idx % args.print_step == 0 or batch_idx == len(train_loader)-1:
                print('[Epoch:%3d/%3d | Batch:%4d/%4d] loss_total: %.4f loss_bcd: %.4f loss_segA: %.4f loss_segB: %.4f loss_contrast: %.4f loss_bscc: %.4f lr: %5f' % 
                    (epoch, args.epochs, batch_idx+1, train_loader.__len__(), losses_total.avg, 
                    losses_bcd.avg, losses_seg_A.avg, losses_seg_B.avg, losses_cont.avg, losses_bscc.avg, get_lr(optimizer))
                )
                
        lr_scheduler.step() 
            
        Log.info('[Training   Epoch:%3d/%3d] loss_total: %.4f loss_bcd: %.4f loss_segA: %.4f loss_segB: %.4f loss_contrast: %.4f loss_bscc: %.4f lr: %f' % 
                (epoch, args.epochs, losses_total.avg, losses_bcd.avg, losses_seg_A.avg, losses_seg_B.avg, losses_cont.avg, losses_bscc.avg, get_lr(optimizer))) 
        
        # validation
        evaluator_bcd.reset()
        evaluator_seg_A.reset()
        evaluator_seg_B.reset()
        evaluator_seg_total.reset()
        
        valosses_seg_A = AverageMeter()
        valosses_seg_B = AverageMeter()
        valosses_bcd = AverageMeter()
        valosses_total = AverageMeter()
            
        model.eval()
        with torch.no_grad():
            for batch_idx, sample in enumerate(val_loader):
                
                img_A, img_B, label_BCD, label_SGA, label_SGB = split_sample(sample, seg_pretrain=args.seg_pretrain)    
                
                imgs = torch.cat([img_A, img_B], dim=1) 
                if args.with_sacl or args.with_bscc:
                    outputs = model(imgs, test=True)   
                else:
                    outputs = model(imgs)
                
                if not args.only_seg:
                    loss_cd = loss_bcd(outputs['BCD'], label_BCD)
                else:
                    loss_cd = torch.tensor(0)
                    
                if not args.only_bcd and (not args.seg_pretrain):
                    loss_seg_A = loss_seg(outputs['seg_A'], label_SGA)
                    loss_seg_B = loss_seg(outputs['seg_B'], label_SGB)
                elif not args.only_bcd and args.seg_pretrain:
                    loss_seg_A = loss_seg(torch.cat([outputs['seg_A'], outputs['seg_B']], dim=0), torch.cat([label_SGA, label_SGB], dim=0))
                    loss_seg_B = torch.tensor(0)
                else:
                    loss_seg_A = torch.tensor(0)
                    loss_seg_B = torch.tensor(0)
                loss = loss_cd + loss_seg_A + loss_seg_B 
                
                valosses_bcd.update(loss_cd.item())    
                valosses_seg_A.update(loss_seg_A.item())
                valosses_seg_B.update(loss_seg_B.item())
                valosses_total.update(loss.item())
                
                if not args.only_seg:
                    pred_bcd = outputs['BCD'].sigmoid().squeeze().cpu().detach().numpy().round().astype('int')
                    evaluator_bcd.add_batch(label_BCD.cpu().numpy().astype('int').squeeze(), pred_bcd)
                
                if not args.only_bcd and args.separate_val_seg:
                    pred_seg_A = torch.argmax(outputs['seg_A'], 1).cpu().detach().numpy().astype('int')
                    evaluator_seg_A.add_batch(label_SGA.cpu().numpy().astype('int'), pred_seg_A)
                    pred_seg_B = torch.argmax(outputs['seg_B'], 1).cpu().detach().numpy().astype('int')
                    evaluator_seg_B.add_batch(label_SGB.cpu().numpy().astype('int'), pred_seg_B)
                elif not args.only_bcd and (not args.separate_val_seg):
                    pred_seg_A = torch.argmax(outputs['seg_A'], 1).cpu().detach().numpy().astype('int')
                    pred_seg_B = torch.argmax(outputs['seg_B'], 1).cpu().detach().numpy().astype('int')
                    pred_seg = np.concatenate([pred_seg_A, pred_seg_B], axis=0)
                    label_seg = torch.cat([label_SGA, label_SGB], dim=0)
                    evaluator_seg_total.add_batch(label_seg.cpu().numpy().astype('int'), pred_seg)
                                    
                if batch_idx % args.print_step == 0 or batch_idx == len(val_loader)-1:
                    print('[Epoch:%3d/%3d | Batch:%4d/%4d] loss_total: %.4f loss_bcd: %.4f loss_segA: %.4f loss_segB: %.4f' %
                        (epoch, args.epochs, batch_idx+1, val_loader.__len__(), valosses_total.avg, 
                         valosses_bcd.avg, valosses_seg_A.avg, valosses_seg_B.avg)
                    ) 
                    
            Log.info('[Validation Epoch:%3d/%3d] loss_total: %.4f loss_bcd: %.4f loss_segA: %.4f loss_segB: %.4f' % 
                    (epoch, args.epochs, valosses_total.avg, valosses_bcd.avg, valosses_seg_A.avg, valosses_seg_B.avg))
            
            if not args.only_seg:         
                OA_bcd = evaluator_bcd.Overall_Accuracy()
                IoU_bcd = evaluator_bcd.Intersection_over_Union()
                F1_bcd = evaluator_bcd.F1_score()
            else:
                OA_bcd = IoU_bcd = F1_bcd = 0
            
            if not args.only_bcd:
                OA_seg_A = evaluator_seg_A.Overall_Accuracy()
                mIoU_seg_A = evaluator_seg_A.Mean_Intersection_over_Union()
                F1_seg_A = evaluator_seg_A.F1_score().mean()
                
                OA_seg_B = evaluator_seg_B.Overall_Accuracy()
                mIoU_seg_B = evaluator_seg_B.Mean_Intersection_over_Union()
                F1_seg_B = evaluator_seg_B.F1_score().mean()
                
                OA_seg_total = evaluator_seg_total.Overall_Accuracy()
                mIoU_seg_total = evaluator_seg_total.Mean_Intersection_over_Union()
                F1_seg_total = evaluator_seg_total.F1_score().mean()
            else:
                OA_seg_A = mIoU_seg_A = F1_seg_A = OA_seg_B = mIoU_seg_B = F1_seg_B = OA_seg_total = mIoU_seg_total = F1_seg_total = 0
            
            Log.info('[Validation Epoch:%3d/%3d] OA_BCD: %.4f IoU_BCD: %.4f F1_BCD: %.4f OA_SEG_A: %.4f mIoU_SEG_A: %.4f F1_SEG_A: %.4f OA_SEG_B: %.4f mIoU_SEG_B: %.4f F1_SEG_B: %.4f OA_SEG_total: %.4f mIoU_SEG_total: %.4f F1_SEG_total: %.4f' % 
                    (epoch, args.epochs, OA_bcd, IoU_bcd, F1_bcd, OA_seg_A, mIoU_seg_A, F1_seg_A,
                     OA_seg_B, mIoU_seg_B, F1_seg_B, OA_seg_total, mIoU_seg_total, F1_seg_total))
        
        if args.separate_val_seg:
            metric_current = IoU_bcd + mIoU_seg_A +  mIoU_seg_B
        else:
            metric_current = IoU_bcd + mIoU_seg_total
        
        if (metric_current > metric_best) or (epoch == 1):
            metric_best_dict = {}
            metric_best_dict["IoU_BCD"] = IoU_bcd
            metric_best_dict["mIoU_SEG_A"] = mIoU_seg_A
            metric_best_dict["mIoU_SEG_B"] = mIoU_seg_B
            metric_best_dict["mIoU_SEG_total"] = mIoU_seg_total
            epoch_best = epoch
            metric_best = metric_current

        if args.epochs > 0:   
            if epoch > 0:
                saver.save_checkpoint({
                    'state_dict': model.state_dict(),
                }, epoch, metric_current)         
            
        print('=> Current metric %.4f Best metric %.4f' % (metric_current, metric_best))
        Log.info('=> Best epoch {} Best metric {}'.format(epoch_best, metric_best_dict))
          
                       
if __name__ == '__main__':   

    floder_list = [
        "./configs/hiucd_mini",
        "./configs/hiucd"
        ]

    for config_floder in floder_list:
        configs_list = os.listdir(config_floder)
        configs_list = [cf for cf in configs_list if cf.endswith('json')]
        for configs_name in configs_list:
            configs_file = os.path.join(config_floder, configs_name)
            args = get_parser_with_args_from_json(configs_file)
            main(args)