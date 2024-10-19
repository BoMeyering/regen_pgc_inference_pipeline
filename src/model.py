# model.py
# BoMeyering 2024

import torch
import cv2
import random
from effdet.config.model_config import efficientdet_model_param_dict
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict
from effdet.efficientdet import HeadNet
import segmentation_models_pytorch as smp
import argparse
import torch

def load_state_dict(device: str):
    path = 'models/mdm_chkpoint/marker_effdet-epoch=19-val_loss=0.113.ckpt'
    state_dict = torch.load(path, map_location=torch.device(device))['state_dict']
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k == 'model.anchors.boxes':
            k = 'anchors.boxes'
        k = k.replace('model.model', 'model')
        new_state_dict[k] = v
    
    return new_state_dict

def create_inference_model(device, num_classes=2, image_size=1024, architecture='tf_efficientdet_d5'):
    config = get_efficientdet_config(architecture)
    config.update({'num_classes': num_classes})
    config.update({'image_size': (image_size, image_size)})
    
    net = EfficientDet(config, pretrained_backbone=False)
    net.class_net = HeadNet(
        config,
        num_outputs=config.num_classes
    )
    
    state_dict = load_state_dict(device)
    
    model = DetBenchPredict(net)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model




def create_smp_model(args: argparse.Namespace) -> torch.nn.Module:
    """Creates an smp Pytorch model

    Args:
        args (argparse.Namespace): The argparse namespace from a config file

    Raises:
        ValueError: If args.model.encoder_name is not listed in smp.encoders.get_encoder_names().
        ValueError: If args.model.model_name does not match any of the specified architectures.

    Returns:
        torch.nn.Module: A model as a pytorch module
    """
    if args.model.encoder_name not in smp.encoders.get_encoder_names():
        raise ValueError(f"Encoder name {args.model.encoder_name} is not one of the accepted encoders. Please select an encoder from {smp.encoders.get_encoder_names()}")
    if args.model.model_name == 'unet':
        model = smp.Unet(
            encoder_name=args.model.encoder_name,
            encoder_depth=args.model.encoder_depth,
            encoder_weights=args.model.encoder_weights,
            in_channels=args.model.in_channels, 
            classes=args.model.num_classes
        )
    elif args.model.model_name == 'unetplusplus':
        model = smp.UnetPlusPlus(
            encoder_name=args.model.encoder_name,
            encoder_depth=args.model.encoder_depth,
            encoder_weights=args.model.encoder_weights,
            in_channels=args.model.in_channels, 
            classes=args.model.num_classes
        )
    elif args.model.model_name == 'manet':
        model = smp.MAnet(
            encoder_name=args.model.encoder_name,
            encoder_depth=args.model.encoder_depth,
            encoder_weights=args.model.encoder_weights,
            in_channels=args.model.in_channels, 
            classes=args.model.num_classes
        )
    elif args.model.model_name == 'linknet':
        model = smp.Linknet(
            encoder_name=args.model.encoder_name,
            encoder_depth=args.model.encoder_depth,
            encoder_weights=args.model.encoder_weights,
            in_channels=args.model.in_channels, 
            classes=args.model.num_classes
        )
    elif args.model.model_name == 'fpn':
        model = smp.FPN(
            encoder_name=args.model.encoder_name,
            encoder_depth=args.model.encoder_depth,
            encoder_weights=args.model.encoder_weights,
            in_channels=args.model.in_channels, 
            classes=args.model.num_classes
        )
    elif args.model.model_name == 'pspnet':
        model = smp.PSPNet(
            encoder_name=args.model.encoder_name,
            encoder_depth=args.model.encoder_depth,
            encoder_weights=args.model.encoder_weights,
            in_channels=args.model.in_channels, 
            classes=args.model.num_classes
        )
    elif args.model.model_name == 'pan':
        model = smp.PAN(
            encoder_name=args.model.encoder_name,
            encoder_weights=args.model.encoder_weights,
            in_channels=args.model.in_channels, 
            classes=args.model.num_classes
        )
    elif args.model.model_name == 'deeplabv3':
        model = smp.DeepLabV3(
            encoder_name=args.model.encoder_name,
            encoder_depth=args.model.encoder_depth,
            encoder_weights=args.model.encoder_weights,
            in_channels=args.model.in_channels, 
            classes=args.model.num_classes
        )
    elif args.model.model_name == 'deeplabv3plus':
        model = smp.DeepLabV3Plus(
            encoder_name=args.model.encoder_name,
            encoder_depth=args.model.encoder_depth,
            encoder_weights=args.model.encoder_weights,
            in_channels=args.model.in_channels, 
            classes=args.model.num_classes
        )
    else:
        raise ValueError(f"args.model.model_name: {args.model.model_name} is not a valid model name for the smp framework. Please select a different architecture from {list(map(str.lower, dir(smp)[:9]))}")

    return model



if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = argparse.Namespace()
    setattr(args, 'model', argparse.Namespace)
    model_args = {
        'model_name': 'deeplabv3plus',
        'num_classes': 8,
        'in_channels': 3,
        'encoder_name': 'efficientnet-b4',
        'encoder_weights': 'imagenet', 
        'encoder_depth': 5,
    }
    for k, v in model_args.items():
        setattr(args.model, k, v)
    model = create_smp_model(args)
 
    # state_dict = torch.load('models/pgc_chkpt/dlv3p_1024_enb4_recall_ce_2024-06-07_00.09.45_epoch_18_2024-06-08_04.28.23', map_location=device)['model_state_dict'] # Baseline no RegenPGC training
    # state_dict = torch.load('models/pgc_chkpt/injection_dlv3p_1024_enb4_recall_ce_2024-06-28_19.35.47_epoch_24_2024-06-30_04.07.56', map_location=device)['model_state_dict'] # Baseline injection
    # state_dict = torch.load('models/pgc_chkpt/fixmatch_dlv3p_1024_enb4_v2_2024-07-03_15.20.41_epoch_1_2024-07-04_00.33.57', map_location=device)['model_state_dict'] # Epoch 1
    # state_dict = torch.load('models/pgc_chkpt/fixmatch_dlv3p_1024_enb4_v2_2024-07-03_15.20.41_epoch_2_2024-07-04_05.15.08', map_location=device)['model_state_dict'] # Epoch 2
    # state_dict = torch.load('models/pgc_chkpt/fixmatch_dlv3p_1024_enb4_v2_2024-07-03_15.20.41_epoch_13_2024-07-06_08.46.53', map_location=device)['model_state_dict'] # Epoch 13
    # state_dict = torch.load('models/pgc_chkpt/fixmatch_dlv3p_1024_enb4_v3_2024-07-09_03.05.35_epoch_13_2024-07-11_22.35.07', map_location=device)['model_state_dict'] # Epoch 13)
    state_dict = torch.load('models/pgc_chkpt/fixmatch_dlv3p_1024_enb4_regenpgc_set_2024-07-12_13.23.20_epoch_18_2024-07-13_04.01.58', map_location=device)['model_state_dict'] # Epoch 18)

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    torch.save(model, 'models/pgc_model.pth')

    model = create_inference_model(torch.device('cuda'))
    torch.save(model, 'models/mdm_model.pth')