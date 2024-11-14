import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import pandas as pd

from dataset.dataset import SECONDDataset, HiUCDDataset, NanjingDataset
from utils.evaluator import SEGEvaluator, BCDEvaluator, SCD_NoChange_Evaluator, SCD_Change_Evaluator
from model import *
from utils.helper import *
from utils.parser import get_parser_with_args_from_json
from loguru import logger
from pprint import pprint
  
 
def split_sample(sample):
    img_A = sample['img_A'].cuda(non_blocking=True)
    img_B = sample['img_B'].cuda(non_blocking=True)
    label_BCD = sample['label_BCD'].cuda(non_blocking=True)
    label_SGA = sample['label_SGA'].cuda(non_blocking=True)
    label_SGB = sample['label_SGB'].cuda(non_blocking=True)   
    return img_A, img_B, label_BCD, label_SGA, label_SGB


def cal_kappa(hist):
    if hist.sum() == 0:
        po = 0
        pe = 1
        kappa = 0
    else:
        po = np.diag(hist).sum() / hist.sum()
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
        if pe == 1:
            kappa = 0
        else:
            kappa = (po - pe) / (1 - pe)
    return kappa


def BCD_PCC_pre(pred_A, pred_B):
    pcc_bcd = 1 - (pred_A == pred_B).astype(np.int8)
    return pcc_bcd


# 计算bcd结果与双时相语义结果矛盾的像素数目
'''
Specifically, a TP pixel
in the BCD evaluation is reclassified into FP if the semantic maps at two
time points have consistent semantics.
'''
def percent_invalid_area(pred_A, pred_B, change_mask, label_BCD):
    right_BCD = (label_BCD != 255).astype(np.int8)
    right_BCD = (label_BCD * change_mask * right_BCD)
    pred_A = pred_A[right_BCD==1]
    pred_B = pred_B[right_BCD==1]
    invalid_area = (pred_A == pred_B).sum()
    return invalid_area


def load_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path) 
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, map)  
    return model


def class2RGB(class_map, change_mask, class_num=6, visual_seg=False):
    if class_num == 6: # SECOND
        CLASS_RGB_VALUES = [[255, 0, 0], [0, 128, 0], [128, 128, 128], [0, 0, 255], 
                            [128, 0, 0], [0, 255, 0]] 
    elif class_num == 9: # Hi-UCD series
        CLASS_RGB_VALUES = [[0, 153, 255], [202, 255, 122], [255, 0, 0], [230, 0, 255], 
                            [255, 230, 0], [255, 181, 197], [0, 255, 230], [175, 122, 255], [0, 255, 0]] 
    elif class_num == 7: # LsSCD
        CLASS_RGB_VALUES = [[255, 181, 197], [235, 0, 0], [255, 255, 0], [0, 100, 200],
                            [250, 230, 160], [0, 100, 0], [255, 170, 0]] 
    h, w = class_map.shape
    rgb = np.zeros([h, w, 3]).astype(np.uint8)
    for i in range(class_num):
        rgb[class_map==i, :] = CLASS_RGB_VALUES[i]
    if not visual_seg:
        unchange_mask = 1 - change_mask
        rgb[unchange_mask==1, :] = 0
    return rgb


def cal_change_type(t1, t2, change_mask, change_label, class_num):
    '''for mIoUsc'''
    from_to = t1 * class_num + t2
    diagonals = [i*class_num + i for i in range(class_num)]
    change_type = from_to - np.searchsorted(diagonals, from_to)
    change_type[change_label == 255] = class_num * (class_num - 1) + 1 # unlabled area equal last class_number + 1
    change_type[change_mask == 0] = class_num * (class_num - 1) # unchange equal the last class_number
    return change_type


def cal_nochange_type(t1, t2, change_mask, change_label, class_num): # 看未变化区域是否一致
    '''for mIoUnc'''
    from_to = t1 * class_num + t2
    diagonals = [i*class_num + i for i in range(class_num)]
    nochange_type = np.searchsorted(diagonals, from_to) # resort 0-classnum-1
    # bcd识别为未变化 但是不符合未变化的from-to类型的等于变化 等同于分配错误的类别 
    # 这里统一分配到了变化类 会影响变化类精度 所以最后计算mean时替换成单独计算的变化类精度   
    nochange_type[~np.isin(from_to, diagonals) & (change_mask == 0)] = class_num # 不在未变类型但是识别为未变化的像素分配变化
    nochange_type[change_label == 255] = class_num + 1
    nochange_type[change_mask == 1] = class_num
    return nochange_type 


VISUAL = False
VISUAL_SEG = False
Log = True
batch_size = 1 if VISUAL else 16
visual_seg_flag = '_seg' if VISUAL_SEG else ''  
              
              
if __name__ == '__main__':
    
    trace = logger.add(os.path.join(r'./results/eval_log.log'))
    config_path = r'./results/config_ClearSCD_hiucd_mini.json'
    checkpoint_path = r'./results/checkpoint.pth.tar'
    infer_output_floder = r'./results/inferences'
    
    args = get_parser_with_args_from_json(config_path) 
    model = get_model(args).cuda()

    if args.dataset == 'SECOND':
        dataset = SECONDDataset(args, split='test')
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        class_num = args.num_segclass
    elif args.dataset == 'HiUCD':
        dataset = HiUCDDataset(args, split='test')
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        class_num = args.num_segclass
    elif args.dataset == 'Nanjing':
        dataset = NanjingDataset(args, split='test')    
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        class_num = args.num_segclass
    
    
    evaluator_seg = SEGEvaluator(class_num**2 + 1) # 从0开始计数，包括无变化类
    evaluator_seg_pair = SEGEvaluator(class_num) 
    evaluator_bcd = SEGEvaluator(2)
    evaluator_pcc_bcd = BCDEvaluator(2) # 
    evaluator_binary_consist = SEGEvaluator(2) # mIoUbc
    evaluator_change_consist = SCD_Change_Evaluator(class_num*(class_num-1)+1) # mIoUsc
    evaluator_nochange_consist = SCD_NoChange_Evaluator(class_num+1) # mIoUnc
                    

    if VISUAL:
        if not os.path.exists(infer_output_floder):
            os.makedirs(infer_output_floder) 
        if not os.path.exists(infer_output_floder + os.sep + 'im1'):
            os.makedirs(infer_output_floder + os.sep + 'im1') 
        if not os.path.exists(infer_output_floder + os.sep + 'im2'):
            os.makedirs(infer_output_floder + os.sep + 'im2') 

    model = load_model(model, checkpoint_path)

    # testing
    model.eval()
    evaluator_seg.reset()
    evaluator_seg_pair.reset()
    evaluator_bcd.reset()
    evaluator_pcc_bcd.reset()
    evaluator_binary_consist.reset()
    evaluator_change_consist.reset()
    evaluator_nochange_consist.reset()
    invalid_area_sum = 0
    right_BCD_area_sum = 0
    loop = tqdm(enumerate(test_loader), total=len(test_loader))
    
    with torch.no_grad():
        for _, sample in loop:
            imgs_A, imgs_B, labels_BCD, labels_A, labels_B = split_sample(sample)    
            imgs = torch.cat([imgs_A, imgs_B], dim=1)
            
            if args.with_cl or args.with_bcl:
                outputs = model(imgs, test=True)
            else:
                outputs = model(imgs)  

            outputs_A = outputs['seg_A'].cpu().detach()
            outputs_B = outputs['seg_B'].cpu().detach()
            
            labels_BCD = labels_BCD.cpu().squeeze().long().detach().numpy()
            labels_A = labels_A.cpu().squeeze().long().detach().numpy() 
            labels_B = labels_B.cpu().squeeze().long().detach().numpy()
            
            preds_A = torch.argmax(outputs_A, dim=1).squeeze().long().numpy()
            preds_B = torch.argmax(outputs_B, dim=1).squeeze().long().numpy()         
            
            # PCC的bcd结果
            pcc_bcd = 1 - (preds_A == preds_B)    
                    
            out_change = outputs['BCD']  
            change_mask = torch.sigmoid(out_change).cpu().detach().squeeze().round().long().numpy()  
            
            # change type的结果
            preds_change_type = cal_change_type(preds_A, preds_B, change_mask, labels_BCD, class_num)
            labels_change_type = cal_change_type(labels_A, labels_B, labels_BCD, labels_BCD, class_num) 
            
            # nochange type的结果
            preds_nochange_type = cal_nochange_type(preds_A, preds_B, change_mask, labels_BCD, class_num)
            labels_nochange_type = cal_nochange_type(labels_A, labels_B, labels_BCD, labels_BCD, class_num)
            
            invalid_area = percent_invalid_area(preds_A, preds_B, change_mask, labels_BCD)             
            right_BCD = (labels_BCD != 255).astype(np.int8)
            right_BCD_area = (labels_BCD * change_mask * right_BCD).sum()        
                            
            if VISUAL:
                if args.dataset == 'HiUCD':
                    labeled_A = (labels_A != class_num)
                    labeled_B = (labels_A != class_num)
                    labeled = labeled_A & labeled_B    
                    change_mask = change_mask * labeled
                img_name = os.path.basename(sample['name'][0])   
                predA_output_path = os.path.join(infer_output_floder, 'im1', img_name)
                pred_A_output = class2RGB(preds_A, change_mask, class_num=class_num, visual_seg=visual_seg_flag)
                pred_A_output = Image.fromarray(pred_A_output).save(predA_output_path)
                
                predB_output_path = os.path.join(infer_output_floder, 'im2', img_name)
                pred_B_output = class2RGB(preds_B, change_mask, class_num=class_num, visual_seg=visual_seg_flag)
                pred_B_output = Image.fromarray(pred_B_output).save(predB_output_path)  
                
            label_scd = labels_A * class_num + labels_B
            preds_scd = preds_A * class_num + preds_B 
            no_labels_area = (labels_A == class_num) | (labels_B == class_num)
            
            if args.dataset == 'SECOND':
                label_scd[no_labels_area] = class_num**2 # 无标注区域为未变化区域
                preds_scd[change_mask == 0] = class_num**2 # 未变化是最后一类
            elif args.dataset == 'HiUCD': # 排除无标注区域和未变化区域                            
                label_scd[no_labels_area] = 999
                dont_care = (labels_A == labels_B) & (label_scd != 999)
                label_scd[dont_care] = class_num**2
                preds_scd[change_mask == 0] = class_num**2
            elif args.dataset == 'Nanjing':
                label_scd[no_labels_area] = 999
                dont_care = (labels_A == labels_B) & (label_scd != 999)
                label_scd[dont_care] = class_num**2
                preds_scd[change_mask == 0] = class_num**2

            seg_pair = np.concatenate([labels_A, labels_B], axis=0) 
            preds_pair = np.concatenate([preds_A, preds_B], axis=0) 
                
            evaluator_seg_pair.add_batch(seg_pair, preds_pair)   
            evaluator_seg.add_batch(label_scd, preds_scd)
            evaluator_bcd.add_batch(labels_BCD, change_mask)
            evaluator_binary_consist.add_batch(change_mask, pcc_bcd)
            evaluator_change_consist.add_batch(labels_change_type, preds_change_type)
            evaluator_nochange_consist.add_batch(labels_nochange_type, preds_nochange_type)

            right_BCD_area_sum += right_BCD_area
            invalid_area_sum += invalid_area
            
        # cal Sek Mask change IoU
        confusion_matrix_seg = evaluator_seg.confusion_matrix
        # 未变化类为0
        OA_scd = np.trace(confusion_matrix_seg) / np.sum(confusion_matrix_seg)
        kappa_trad = cal_kappa(confusion_matrix_seg) # kappa_trad
        confusion_matrix_seg[class_num**2, class_num**2] = 0 # except no change  

        kappa_nuc = cal_kappa(confusion_matrix_seg) # kappa_nuc
        IoU_mean = evaluator_bcd.Mean_Intersection_over_Union()
        IoU_change = evaluator_bcd.Intersection_over_Union()[1] # BCD IOU_trad
        IoU_nochange = evaluator_bcd.Intersection_over_Union()[0]
        F1_change = evaluator_bcd.F1_score()[1] # BCD F1_trad
        Sek = (kappa_nuc * math.exp(IoU_change)) / math.e # SeK
        PCC_BCD_IoU_change = evaluator_pcc_bcd.Intersection_over_Union()
        PCC_BCD_F1_change = evaluator_pcc_bcd.F1_score()
        
        # 修正变化类IoU
        evaluator_nochange_consist.MIoU_pre()
        evaluator_nochange_consist.MIoU_list[-1] = IoU_change
        
        evaluator_bcd.pre_cal = False
        evaluator_bcd.confusion_matrix[1, 1] = evaluator_bcd.confusion_matrix[1, 1] - invalid_area_sum # TP
        evaluator_bcd.confusion_matrix[0, 1] = evaluator_bcd.confusion_matrix[0, 1] + invalid_area_sum # FP
        IoU_change_mod = evaluator_bcd.Intersection_over_Union()[1] # BCD F1_mod
        F1_change_mod = evaluator_bcd.F1_score()[1] # BCD F1_mod
        Sek_mod = (kappa_nuc * math.exp(IoU_change_mod)) / math.e # Sek_mod
        
        seg_pair_IoU = evaluator_seg_pair.Mean_Intersection_over_Union()
        seg_pair_F1 = evaluator_seg_pair.F1_score().mean()
        
        # 计算binary consistency
        binary_consistency = evaluator_binary_consist.Mean_Intersection_over_Union()  
        # 计算change consistency
        change_consistency = evaluator_change_consist.Mean_Intersection_over_Union()
        # 计算nochange consistency
        nochange_consistency = evaluator_nochange_consist.Mean_Intersection_over_Union()
                        
        res_dict = {}
        res_dict['BCD_IoU'] = np.round(IoU_change*100, 2)
        res_dict['BCD_F1'] = np.round(F1_change*100, 2)
        res_dict['BCD_IoU_mod'] = np.round(IoU_change_mod*100, 2)
        res_dict['BCD_F1_mod'] = np.round(F1_change_mod*100, 2)
        
        res_dict['SEG_mIoU'] = np.round(seg_pair_IoU*100, 2)
        res_dict['SEG_mF1'] = np.round(seg_pair_F1*100, 2)
        
        res_dict['SCD_Sek'] = np.round(Sek*100, 2)
        res_dict['SCD_Sek_mod'] = np.round(Sek_mod*100, 2)
        res_dict['Kappa'] = np.round(kappa_trad*100, 2)
        
        res_dict['Overall_Score'] = np.around(0.3 * res_dict['BCD_IoU'] + 0.7 * res_dict['SCD_Sek'], 2)
        res_dict['Overall_Score_mod'] = np.around(0.3 * res_dict['BCD_IoU_mod'] + 0.7 * res_dict['SCD_Sek_mod'], 2)
        
        res_dict['binary_consistency'] = np.round(binary_consistency*100, 2)
        res_dict['change_consistency'] = np.round(change_consistency*100, 2)
        res_dict['nochange_consistency'] = np.round(nochange_consistency*100, 2)
        
        pprint(res_dict, indent=4)
            
        if Log:
            logger.info(res_dict)          
    logger.remove(trace)