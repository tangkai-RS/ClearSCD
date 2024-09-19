import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from loss.losses import BSCCLoss
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.decoders.fpn.decoder import FPNDecoder
from segmentation_models_pytorch.encoders.resnet import (ResNetEncoder, resnet_encoders)
from segmentation_models_pytorch.encoders.efficientnet import (EfficientNetEncoder, efficient_net_encoders)
from einops import repeat, rearrange


class ChangeDetectionHead(nn.Module):
    '''CVAPS module'''
    def __init__(
        self,
        in_channels = 128,
        inner_channels = 16,
        num_convs = 4,
        upsampling = 4,
        dilation = 1,
        fusion = 'diff',
    ):
        super(ChangeDetectionHead, self).__init__()
        if fusion == 'diff':
            in_channels = in_channels
            inner_channels = in_channels
        elif fusion == 'concat':
            in_channels = in_channels * 2
            inner_channels = in_channels
        layers = []
        if num_convs > 0:
            layers = [
                nn.modules.Sequential(
                    nn.modules.Conv2d(in_channels, inner_channels, 3, 1, 1, dilation=dilation),
                    nn.modules.BatchNorm2d(inner_channels),
                    nn.modules.ReLU(True),
                )
            ]
            if num_convs >  1:
                layers += [
                    nn.modules.Sequential(
                        nn.modules.Conv2d(inner_channels, inner_channels, 3, 1, 1, dilation=dilation),
                        nn.modules.BatchNorm2d(inner_channels),
                        nn.modules.ReLU(True),
                    )
                    for _ in range(num_convs - 1)
                ]

        cls_layer = nn.modules.Conv2d(inner_channels, 1, 3, 1, 1)
        layers.append(cls_layer) 
        self.convs = nn.modules.Sequential(*layers)
        
        self.upsampling_scale_factor = upsampling
        self.upsampling = nn.modules.UpsamplingBilinear2d(scale_factor=upsampling) if self.upsampling_scale_factor > 1 else nn.Identity()
        
    def forward(self, x, with_bscc):  
        x = self.convs(x)
        x_upsampling = self.upsampling(x)
        if with_bscc:
            return x, x_upsampling
        elif not with_bscc:
            return torch.tensor(0), x_upsampling


class ProjectionHead(nn.Module):
    def __init__(self, in_channels=128, proj_dim=64):
        super(ProjectionHead, self).__init__()
        
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, proj_dim, kernel_size=1),
            nn.BatchNorm2d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(proj_dim, in_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.proj(x) 
       

class ClearSCD(nn.Module):
    def __init__(self, args):
        super(ClearSCD, self).__init__()
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.bscc = BSCCLoss()
        if 'res' in self.args.backbone:
            encoder_params = resnet_encoders[args.backbone]['params']
            self.encoder = ResNetEncoder(**encoder_params)
        elif 'eff' in self.args.backbone:
            encoder_params = efficient_net_encoders[args.backbone]['params']
            self.encoder = EfficientNetEncoder(**encoder_params)
        
        self.decoder = FPNDecoder(
            encoder_channels = encoder_params['out_channels'],
            pyramid_channels = 256,
            segmentation_channels = 128,
        )

        upsampling = 1 if args.downsample_seg else 4
        self.head_seg = SegmentationHead(
            in_channels = 128,
            out_channels = self.args.num_segclass,
            upsampling = upsampling   
        ) 
        
        upsampling_bcd = 4 if args.downsample_seg else 1
        self.head_bcd = ChangeDetectionHead(
            in_channels = self.args.num_segclass,
            inner_channels = self.args.num_segclass,
            num_convs = self.args.bcd_convs_num,
            upsampling = upsampling_bcd,
            fusion = self.args.fusion
        ) 
        
        self.proj_head = ProjectionHead(
            in_channels = 128,
            proj_dim = 32
        )
        
        self.pool = nn.AvgPool2d(kernel_size=4, stride=4)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=4)
                
        for i in range(self.args.num_segclass):
            self.register_buffer("queue_A" + str(i), torch.randn(self.args.proj_dim, self.args.queue_len))
            self.register_buffer("ptr_A" + str(i), torch.zeros(1, dtype=torch.long))
            exec("self.queue_A" + str(i) + '=' + 'nn.functional.normalize(' + "self.queue_A" + str(i) + ', dim=0)')
            
            self.register_buffer("queue_B" + str(i), torch.randn(self.args.proj_dim, self.args.queue_len))
            self.register_buffer("ptr_B" + str(i), torch.zeros(1, dtype=torch.long))
            exec("self.queue_B" + str(i) + '=' + 'nn.functional.normalize(' + "self.queue_B" + str(i) + ', dim=0)')     
        
        if self.args.pretrained:            
            self._init_weighets()
        else:
            self._init_weighets_kaiming(self.encoder, self.decoder, self.head_bcd, self.head_seg, self.proj_head)
        
    def _init_weighets(self):
        if self.args.backbone == 'resnet18':
            encoder_pred = torchvision.models.resnet18(weights='ResNet18_Weights.DEFAULT')
        elif self.args.backbone == 'resnet34':
            encoder_pred = torchvision.models.resnet34(weights='ResNet34_Weights.DEFAULT')   
        elif self.args.backbone == 'resnet50':    
            encoder_pred = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')  
        elif self.args.backbone == 'efficientnet-b0':
            encoder_pred = torchvision.models.efficientnet_b0(weights='EfficientNet_B0_Weights.IMAGENET1K_V1')  
        elif self.args.backbone == 'efficientnet-b2':
            encoder_pred = torchvision.models.efficientnet_b2(weights='EfficientNet_B2_Weights.IMAGENET1K_V1')  
                       
        pretrained_dict = encoder_pred.state_dict()
        encoder_dict = self.encoder.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
        encoder_dict.update(pretrained_dict)   
        
    def _init_weighets_kaiming(*models):
        for model in models:
            for module in model.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_() 
                    
    def _dequeue_and_enqueue(self, keys, class_vals, id='A or B'):
        '''区分类别向queue中存放每个类别的特征'''     
        for i in range(self.args.num_segclass):
            if i not in class_vals:
                continue
            else:
                idx = (class_vals==i)
                if torch.sum(idx) < self.args.enqueue_len:
                    continue
                keys = F.normalize(keys, dim=0, p=2)
                keys_class = keys[:, idx]
                ptr = int(eval("self.ptr_" + id + str(i)))
                eval("self.queue_" + id + str(i))[:, ptr : ptr + self.args.enqueue_len] = keys_class
                ptr = (ptr + self.args.enqueue_len) % self.args.queue_len
                eval("self.ptr_" + id + str(i))[0] = ptr 
        
    @torch.no_grad()      
    def construct_key_topK(self, fea, pred, label):
        '''根据label和pred在每个类别中抽取置信度最高的像素特征'''
        bs = fea.shape[0]
        if not self.args.downsample_seg:
            pred = self.pool(pred).detach() 
        else:
            pred = pred.detach()
        label = label.view(bs, -1).view(-1)
        
        pred_soft = self.softmax(pred)
        pred = pred_soft.max(1)[1].squeeze().view(bs, -1).view(-1)
        pred_soft = pred_soft.max(1)[0].squeeze().view(bs, -1).view(-1)
        
        fea = fea.squeeze()
        fea = fea.reshape(bs, self.args.proj_dim, -1).permute(1, 0, 2).reshape(self.args.proj_dim, -1)
        
        intersect = (pred == label)
        class_cur = pred.clone()[intersect]
        class_cur = torch.unique(class_cur)
        
        init = 0
        while True:
            idx = (pred==class_cur[init]) & (label==class_cur[init])
            pred_right_soft = pred_soft[idx]
            new_fea = fea[:, idx]
            pred_right_sort, sort_idx = torch.sort(pred_right_soft, descending=True)
            new_fea = new_fea[:, sort_idx]
            if torch.sum(pred_right_sort > self.args.confidence) != 0:
                new_fea = new_fea[:, pred_right_sort > self.args.confidence]
                if new_fea.shape[1] > self.args.enqueue_len:
                    new_fea = new_fea[:, 0:self.args.enqueue_len]
                class_val = repeat(torch.tensor([class_cur[init]]), 'n -> (repeat n)', repeat=new_fea.shape[1])
                init += 1
                break
            elif (init + 1) != class_cur.size().numel(): 
                init += 1
            else:
                class_val = torch.tensor([self.args.num_segclass]).cuda()
                break
            
        if (init) != class_cur.size().numel():          
            for i in class_cur[init:]:
                if (i < self.args.num_segclass):
                    idx = (pred==i) & (label==i)
                    pred_right_soft = pred_soft[idx]
                    class_fea = fea[:, idx]
                    pred_right_sort, sort_idx = torch.sort(pred_right_soft, descending=True) 
                    class_fea = class_fea[:, sort_idx]
                    if torch.sum(pred_right_sort > self.args.confidence) != 0:
                        class_fea = class_fea[:, pred_right_sort > self.args.confidence]
                        if class_fea.shape[1] > self.args.enqueue_len:
                            class_fea = class_fea[:, 0:self.args.enqueue_len]
                        class_now = repeat(torch.tensor([i]), 'n -> (repeat n)', repeat=class_fea.shape[1])
                        class_val = torch.cat([class_val, class_now], dim=0)
                        new_fea = torch.cat((new_fea, class_fea), dim=1)
                    else:
                        continue
            return new_fea, class_val.cuda()
        else:
            class_val = torch.tensor([self.args.num_segclass]).cuda()
            return 0, class_val

    @torch.no_grad()      
    def construct_key_random(self, fea, label):
        '''根据label每个类别随机抽取k个像素'''
        bs = fea.shape[0]
        label = label.view(bs, -1).view(-1)
               
        fea = fea.squeeze()
        fea = fea.reshape(bs, self.args.proj_dim, -1).permute(1, 0, 2).reshape(self.args.proj_dim, -1)
        
        class_cur = torch.unique(label)
        
        init = 0
        idx = (label==class_cur[init])
        new_fea = fea[:, idx]
        random_idx = torch.randperm(new_fea.shape[1])
        new_fea = new_fea[:, random_idx]
        if new_fea.shape[1] > self.args.enqueue_len:
            new_fea = new_fea[:, 0:self.args.enqueue_len]
        class_val = repeat(torch.tensor([class_cur[init]]), 'n -> (repeat n)', repeat=new_fea.shape[1])
            
        if (init + 1) != class_cur.size().numel():          
            for i in class_cur[init+1:]:
                if (i < self.args.num_segclass):
                    idx = (label==i)
                    class_fea = fea[:, idx]
                    random_idx = torch.randperm(class_fea.shape[1])
                    class_fea = class_fea[:, random_idx]
                    if class_fea.shape[1] > self.args.enqueue_len:
                        class_fea = class_fea[:, 0:self.args.enqueue_len]
                    class_now = repeat(torch.tensor([i]), 'n -> (repeat n)', repeat=class_fea.shape[1])
                    class_val = torch.cat([class_val, class_now], dim=0)
                    new_fea = torch.cat((new_fea, class_fea), dim=1)
            return new_fea, class_val.cuda()
 
    def extract_pixel_random(self, fea, label):
        '''根据label每个类别随机抽取k个像素'''
        bs = fea.shape[0]
        label = label.view(bs, -1).view(-1)
                
        fea = fea.squeeze()
        fea = fea.reshape(bs, self.args.proj_dim, -1).permute(1, 0, 2).reshape(self.args.proj_dim, -1)
        
        class_cur = torch.unique(label)
        
        init = 0
        idx = (label==class_cur[init])
        new_fea = fea[:, idx]
        random_idx = torch.randperm(new_fea.shape[1])
        new_fea = new_fea[:, random_idx]
        if new_fea.shape[1] > self.args.sample_num:
            new_fea = new_fea[:, 0:self.args.sample_num]
        class_val = repeat(torch.tensor([class_cur[init]]), 'n -> (repeat n)', repeat=new_fea.shape[1])
            
        if (init + 1) != class_cur.size().numel():          
            for i in class_cur[init+1:]:
                if (i < self.args.num_segclass):
                    idx = (label==i)
                    class_fea = fea[:, idx]
                    random_idx = torch.randperm(class_fea.shape[1])
                    class_fea = class_fea[:, random_idx]
                    if class_fea.shape[1] > self.args.sample_num:
                        class_fea = class_fea[:, 0:self.args.sample_num]
                    class_now = repeat(torch.tensor([i]), 'n -> (repeat n)', repeat=class_fea.shape[1])
                    class_val = torch.cat([class_val, class_now], dim=0)
                    new_fea = torch.cat((new_fea, class_fea), dim=1)
            return new_fea, class_val.cuda()
    
    def construct_query_region_hard(self, fea, pred, label):
        '''根据label和预测的每个类别软概率反加权得到当前batch的每个类别中心 hard样本的权重更大'''
        bs = fea.shape[0]
        if not self.args.downsample_seg:
            pred = self.pool(pred).detach() # B C H W
        else:
            pred = pred.detach()
        label = label.view(bs, -1) # B H*W
        
        pred_soft = self.softmax(pred).view(bs, self.args.num_segclass, -1) # B C H*W
        
        fea = fea.squeeze()
        fea = fea.view(bs, self.args.proj_dim, -1).permute(1, 0, 2) # C B H*W
        val = torch.unique(label).long()
        
        idx = (label==val[0]) # B H*W
        pred_soft_class = 1 - pred_soft[:, val[0], :] # B H*W 
        new_fea = fea[:, idx] * pred_soft_class[idx].unsqueeze(0)
        weight = pred_soft_class[idx].sum()
        new_fea = new_fea.sum(1) / weight
        new_fea = new_fea.unsqueeze(0)
            
        for i in val[1:]:
            if (i < self.args.num_segclass):
                idx = (label==i)
                pred_soft_class = 1 - pred_soft[:, i, :]
                class_fea = fea[:, idx] * pred_soft_class[idx].unsqueeze(0)
                weight = pred_soft_class[idx].sum()
                class_fea = class_fea.sum(1) / weight
                class_fea = class_fea.unsqueeze(0)
                new_fea = torch.cat((new_fea, class_fea), dim=0)
        val = torch.tensor([i for i in val if i < self.args.num_segclass])
        return new_fea, val.cuda()

    def construct_region_query_mean(self, fea, label):
        '''根据label将当前batch的每一类别feature简单avg pooling'''
        bs = fea.shape[0]
        label = label.detach().squeeze().view(bs, -1)  
        val = torch.unique(label)
        fea = fea.squeeze()
        fea = fea.view(bs, self.args.proj_dim, -1).permute(1, 0, 2)
    
        new_fea = fea[:, label==val[0]].mean(1).unsqueeze(0)
        for i in val[1:]:
            if (i < self.args.num_segclass):
                class_fea = fea[:, label==i].mean(1).unsqueeze(0)
                new_fea = torch.cat((new_fea, class_fea), dim=0)
        val = torch.tensor([i for i in val if i < self.args.num_segclass])
        return new_fea, val.cuda()
    
    def _compute_infoNCE_loss(self, l_pos, l_neg, N):
        '''l_pos: 1 queuelen * 2  l_neg: 1 class * queuelen'''
        l_pos = l_pos.transpose(1, 0) # queue_len 1
        l_neg = repeat(l_neg, 'c n -> (repeat c) n', repeat=N) # 2*queue_len queue_len*(classnum-1)
        
        logits = torch.cat((l_pos, l_neg), dim=1) # 2*queue_len queue_len*(classnum-1) + 1
        logits /= self.args.temperature
        labels = torch.zeros((logits.shape[0]), dtype=torch.long).cuda()
        return self.criterion(logits, labels)
    
    def _compute_sim_loss(self, l_pos, l_neg):
        '''l_pos: 1 queuelen * 2  l_neg: 1 (class-1) * queuelen'''
        similarity_pos = (1 - l_pos).mean(1) # [0 2]
        similarity_neg = (1 + l_neg).mean(1) # [0 2]   
        return (similarity_pos + similarity_neg) / 2 # [0 2]
    
    def local_contrast(self, proj_feat_A, proj_feat_B, seg_A, seg_B, label_A, label_B):
        contrast_infoNCE_loss = 0
        queries_A_pixels, q_class_val_A = self.extract_pixel_random(proj_feat_A, label_A)
        queries_B_pixels, q_class_val_B = self.extract_pixel_random(proj_feat_B, label_B) 
        
        queries_pixels = torch.cat([queries_A_pixels, queries_B_pixels], dim=1)
        queries_class_val = torch.cat([q_class_val_A, q_class_val_B], dim=0)
        
        keys_A_pixels, k_class_val_A = self.extract_pixel_random(seg_A, label_A)
        keys_A_pixels = F.normalize(keys_A_pixels, dim=0, p=2)
        keys_B_pixels, k_class_val_B = self.extract_pixel_random(seg_B, label_B)
        keys_B_pixels = F.normalize(keys_B_pixels, dim=0, p=2)
        
        keys_pixels = torch.cat([keys_A_pixels, keys_B_pixels], dim=1)
        keys_class_val = torch.cat([k_class_val_A, k_class_val_B], dim=0)
        
        for i in range(self.args.num_segclass):
            if i in queries_class_val:
                idx_anchor = (queries_class_val==i)
                anchor_pixels = queries_pixels[:, idx_anchor]
                anchor = torch.mean(anchor_pixels, dim=1)
                anchor = F.normalize(anchor, dim=0, p=2)
                idx_pos = (keys_class_val==i)
                pos_pixels = keys_pixels[:, idx_pos]
                pos_sim = anchor.unsqueeze(0) @ pos_pixels
                
                idx_neg = (keys_class_val!=i)
                neg_pixels = keys_pixels[:, idx_neg]
                neg_sim = anchor.unsqueeze(0) @ neg_pixels
                contrast_infoNCE_loss += self._compute_infoNCE_loss(pos_sim, neg_sim, pos_pixels.shape[1])
        mean_size = torch.unique(queries_class_val).size().numel()
        contrast_infoNCE_loss = contrast_infoNCE_loss * self.args.contrast_weight / mean_size  
        return contrast_infoNCE_loss
    
    def global_contrast(self, seg_A, seg_B, logits_A, logits_B, label_A, label_B, proj_feat_A, proj_feat_B, warmup):
        if self.args.representive_sample_key:   
            keys_A, vals_kA = self.construct_key_topK(seg_A, logits_A, label_A)
            keys_B, vals_kB = self.construct_key_topK(seg_B, logits_B, label_B)
        else:
            keys_A, vals_kA = self.construct_key_random(seg_A, label_A)       
            keys_B, vals_kB = self.construct_key_random(seg_B, label_B)               
        
        if not warmup:
            contrast_sim_loss_A = 0    
            contrast_sim_loss_B = 0
            if self.args.hard_sampling:                         
                queries_A, vals_qA = self.construct_query_region_hard(proj_feat_A, logits_A, label_A)
                queries_A = F.normalize(queries_A, dim=1, p=2)
                queries_B, vals_qB = self.construct_query_region_hard(proj_feat_B, logits_B, label_B)
                queries_B = F.normalize(queries_B, dim=1, p=2)               
            else:
                queries_A, vals_qA = self.construct_region_query_mean(proj_feat_A, label_A)
                queries_A = F.normalize(queries_A, dim=1, p=2)                    
                queries_B, vals_qB = self.construct_region_query_mean(proj_feat_B, label_B)   
                queries_B = F.normalize(queries_B, dim=1, p=2)                      
            
            # cal contrastive learning loss for AA BB AB BA
            for cls_ind in range(self.args.num_segclass):
                if cls_ind in vals_qA:
                    query_A = queries_A[list(vals_qA).index(cls_ind)]
                    l_pos_AA = query_A.unsqueeze(0) @ eval("self.queue_A" + str(cls_ind)).clone().detach() # 1 queue_len
                    l_pos_AB = query_A.unsqueeze(0) @ eval("self.queue_B" + str(cls_ind)).clone().detach()   
                    all_ind = [m for m in range(self.args.num_segclass)]
                    neg_A = []
                    tmp = all_ind.copy()
                    tmp.remove(cls_ind)
                    for cls_ind2 in tmp:
                        neg_A.append(eval("self.queue_A" + str(cls_ind2)).clone().detach()) # 1 queue_len 
                        neg_A.append(eval("self.queue_B" + str(cls_ind2)).clone().detach())
                    neg_A = torch.stack(neg_A, dim=0) # 2 * (class_num - 1)  C queue_len 
                    neg_A = rearrange(neg_A, 'classnum channel queuelen -> channel (classnum queuelen)') # proj_dim 2560
                    l_pos_A = torch.cat([l_pos_AA, l_pos_AB], dim=1) # 1 queue_len * 2
                    l_neg_A = query_A.unsqueeze(0) @ neg_A # 1 class * queuelen 2560
                    contrast_sim_loss_A += self._compute_sim_loss(l_pos_A, l_neg_A)                            
            
            for cls_ind in range(self.args.num_segclass):
                if cls_ind in vals_qB:
                    query_B = queries_B[list(vals_qB).index(cls_ind)]
                    l_pos_BB = query_B.unsqueeze(0) @ eval("self.queue_B" + str(cls_ind)).clone().detach()     
                    l_pos_BA = query_B.unsqueeze(0) @ eval("self.queue_A" + str(cls_ind)).clone().detach()   
                    all_ind = [m for m in range(self.args.num_segclass)]
                    neg_B = []
                    tmp = all_ind.copy()
                    tmp.remove(cls_ind)
                    for cls_ind2 in tmp:
                        neg_B.append(eval("self.queue_B" + str(cls_ind2)).clone().detach())
                        neg_B.append(eval("self.queue_A" + str(cls_ind2)).clone().detach())
                    neg_B = torch.stack(neg_B, dim=0)
                    neg_B = rearrange(neg_B, 'classnum channel queuelen -> channel (classnum queuelen)')
                    l_pos_B = torch.cat([l_pos_BB, l_pos_BA], dim=1)
                    l_neg_B = query_B.unsqueeze(0) @ neg_B
                    contrast_sim_loss_B += self._compute_sim_loss(l_pos_B, l_neg_B)      
                
            contrast_sim_loss = contrast_sim_loss_A / (2*vals_qA.size(dim=0)) + contrast_sim_loss_B / (2*vals_qB.size(dim=0))
            self._dequeue_and_enqueue(keys_A, vals_kA, id='A')
            self._dequeue_and_enqueue(keys_B, vals_kB, id='B')    
            return contrast_sim_loss
        else:
            self._dequeue_and_enqueue(keys_A, vals_kA, id='A')
            self._dequeue_and_enqueue(keys_B, vals_kB, id='B')
            return 0
               
    def forward(self, imgs, label_A=None, label_B=None, label_BCD=None, test=False, warmup=True):
        img_A = imgs[:, 0:3, :, :]
        img_B = imgs[:, 3::, :, :]      
           
        features_A = self.encoder(img_A)
        features_B = self.encoder(img_B)

        seg_A = self.decoder(*features_A)
        seg_B = self.decoder(*features_B)
        
        # semantic segmentation
        if not self.args.only_bcd:
            logits_A = self.head_seg(seg_A)
            logits_B = self.head_seg(seg_B)
        elif self.args.only_bcd:
            logits_A = None
            logits_B = None
            
        # change detection
        if self.args.only_seg:
            logits_BCD = None
        elif not self.args.only_seg:
            if self.args.fusion == 'diff':
                logits_AB = torch.abs(logits_A - logits_B)
            elif self.args.fusion == 'concat':
                logits_AB = torch.concat([logits_A, logits_B], dim=1)
            # deep CVAPS
            bcd_mask, logits_BCD = self.head_bcd(logits_AB, with_bscc=self.args.with_bscc)
        
        outputs = {}
        # semantic augmentation contrastive learning (SACL)
        if self.args.with_sacl and (not test):
            label_A = F.interpolate(label_A.unsqueeze(1).float(), size=(self.args.size // 4, self.args.size // 4), mode='nearest').squeeze()
            label_B = F.interpolate(label_B.unsqueeze(1).float(), size=(self.args.size // 4, self.args.size // 4), mode='nearest').squeeze()
            
            # project features
            proj_feat_A = self.proj_head(seg_A)
            proj_feat_B = self.proj_head(seg_B)
            
            contrast_sim_loss = 0
            contrast_infoNCE_loss = 0
            # SACL at local level
            if self.args.local_contrast:
                contrast_infoNCE_loss = self.local_contrast(proj_feat_A, proj_feat_B, seg_A, seg_B, label_A, label_B)
            # SACL at global level
            if self.args.global_contrast:
                contrast_sim_loss = self.global_contrast(seg_A, seg_B, logits_A, logits_B, label_A, label_B, proj_feat_A, proj_feat_B, warmup)

            contrast_loss = contrast_sim_loss + contrast_infoNCE_loss 
            outputs['contrast_loss'] = contrast_loss 
        
        if self.args.downsample_seg:
            logits_A = self.upsampling(logits_A)
            logits_B = self.upsampling(logits_B)

        # bscc loss
        if self.args.with_bscc and (not test):
            seg_diff = torch.abs(seg_A - seg_B)
            if self.args.bscc_with_label:
                bcd_mask = F.interpolate(label_BCD.unsqueeze(1).float(), size=(self.args.size // 4, self.args.size // 4), mode='nearest')
                bcd_true_mask = F.interpolate(label_BCD.unsqueeze(1).float(), size=(self.args.size // 4, self.args.size // 4), mode='nearest')
                outputs['bscc_loss'] = self.bscc(seg_diff, bcd_mask, bcd_true_mask) * self.args.bscc_weight
            else:
                bcd_mask = self.sigmoid(bcd_mask)
                # bcd_mask = torch.round(bcd_mask)
                outputs['bscc_loss'] = self.bscc(seg_diff, bcd_mask, torch.round(bcd_mask)) * self.args.bscc_weight
        else: 
            outputs['bscc_loss'] = torch.tensor(0)
        
        outputs['seg_A'] = logits_A
        outputs['seg_B'] = logits_B
        outputs['BCD'] = logits_BCD     
          
        return outputs
   
        
if __name__ == '__main__':
    pass       
