# Reproduced version according to the original paper description "ChangeMask: Deep multi-task encoder-transformer-decoder architecture for semantic change detection"
import torch
import torch.nn as nn
import torchvision

from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.encoders.efficientnet import (EfficientNetEncoder, efficient_net_encoders)


class Squeeze2(nn.Module):
    def forward(self, x):
        return x.squeeze(dim=2)


class TSTBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super().__init__()        
        self.block = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, [2, kernel_size, kernel_size], stride=1, padding=(0, kernel_size//2, kernel_size//2), bias=False),
            Squeeze2(),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )
        
    def forward(self, x):
        x = self.block(x)
        return x


class TST(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TST, self).__init__()   
        
        self.tst_list = nn.ModuleList([TSTBlock(in_channel, out_channel) for in_channel, out_channel in zip(in_channels, out_channels)])
        
    def forward(self, features_A, features_B):
        features_AB = [tst(torch.stack([fa, fb], dim=2)) for tst, fa, fb in zip(self.tst_list, features_A, features_B)]
        features_BA = [tst(torch.stack([fb, fa], dim=2)) for tst, fb, fa in zip(self.tst_list, features_B, features_A)]
        tst_features = [fab * fba for fab, fba in zip(features_AB, features_BA)]
        return tst_features
    

class ChangeMask(nn.Module):
    def __init__(self, args):
        super(ChangeMask, self).__init__()
        self.seg_pretrain = args.seg_pretrain
        encoder_params = efficient_net_encoders['efficientnet-b0']['params']
        self.encoder = EfficientNetEncoder(**encoder_params)
        
        self.seg_decoder = UnetDecoder(
            encoder_channels = encoder_params['out_channels'],
            decoder_channels = (256, 128, 64, 32, 16),
            n_blocks = 5
        )

        self.bcd_decoder = UnetDecoder(
            encoder_channels = encoder_params['out_channels'],
            decoder_channels = (256, 128, 64, 32, 16),
            n_blocks = 5
        )
        
        self.bcd_head = SegmentationHead(in_channels=16, out_channels=1, kernel_size=1)
        self.seg_head = SegmentationHead(in_channels=16, out_channels=args.num_segclass, kernel_size=1)
        
        self.tst = TST(
            in_channels = encoder_params['out_channels'],
            out_channels = encoder_params['out_channels'],
        )
        
        if args.pretrained:
            self._init_weighets()
            
    def _init_weighets(self):
        efficientnet_b0 = torchvision.models.efficientnet_b0(weights='EfficientNet_B0_Weights.IMAGENET1K_V1')                  
        pretrained_dict = efficientnet_b0.state_dict()
        encoder_dict = self.encoder.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
        encoder_dict.update(pretrained_dict)    
                         
    def forward(self, imgs):
        img_A = imgs[:, 0:3, :, :]
        img_B = imgs[:, 3::, :, :]
        
        features_A = self.encoder(img_A)
        features_B = self.encoder(img_B)

        tst_features = self.tst(features_A, features_B)
        logits_BCD = self.bcd_decoder(*tst_features)
        logits_BCD = self.bcd_head(logits_BCD)
            
        seg_A = self.seg_decoder(*features_A)
        seg_B = self.seg_decoder(*features_B)
        logits_A = self.seg_head(seg_A)
        logits_B = self.seg_head(seg_B)
        
        outputs = {}
        outputs['bcl_loss'] = torch.tensor(0)
        outputs['seg_A'] = logits_A
        outputs['seg_B'] = logits_B
        outputs['BCD'] = logits_BCD     
        return outputs    


if __name__ == '__main__': 
    import torch
    
    class args():
        pass
    args.seg_pretrain = False
    args.pretrained = True
    args.num_segclass = 9
    
    model = ChangeMask(args).cuda()
    inputs = torch.randn([8, 6, 512, 512]).cuda()
    label_A = torch.randint(0, 5, size=[8, 512, 512]).cuda()
    label_B = torch.randint(0, 5, size=[8, 512, 512]).cuda()
    
    outputs = model(inputs)   
