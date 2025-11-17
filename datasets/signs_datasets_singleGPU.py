## -*- coding: utf-8 -*-
import os
import sys
import torch
import torch.distributed as dist

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT_DIR = os.path.abspath('/home/wtingting/Downloads/Mamba-code/RFFMamba')
sys.path.append(ROOT_DIR)

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from datasets.gtsrb_dataset import GTSRB
from datasets.btsc_dataset import BelgiumTS
from datasets.ctsd_dataset import CTSD

from timm.data import Mixup

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from torchvision.transforms import InterpolationMode
from timm.data import create_transform
    
def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR
        

def init_dataset(config, args):
    resize_im = config.DATA.IMG_SIZE > 32 # 224
    if args.dataset == 'GTSRB':
    #     transform = create_transform(
    #         input_size=config.DATA.IMG_SIZE,
    #         is_training=True,
    #         color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
    #         auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
    #         re_prob=config.AUG.REPROB,
    #         re_mode=config.AUG.REMODE,
    #         re_count=config.AUG.RECOUNT,
    #         interpolation=config.DATA.INTERPOLATION,
    #     )
    #     if not resize_im:
    #         # replace RandomResizedCropAndInterpolation with
    #         # RandomCrop
    #         transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
    #     transform_train = transform

    #     transform_val = create_transform(
    #         input_size=config.DATA.IMG_SIZE,
    #         is_training=False,
    #         interpolation=config.DATA.INTERPOLATION,
    #         mean=IMAGENET_DEFAULT_MEAN,
    #         std=IMAGENET_DEFAULT_STD,
    #     )    
        # Create Transform
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=(0.8), contrast=(1, 1)),
            transforms.Resize([config.DATA.IMG_SIZE, config.DATA.IMG_SIZE]),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

        ])
        
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Resize([config.DATA.IMG_SIZE, config.DATA.IMG_SIZE]),
        ])

        # Create Datasets
        dir = './datasets/signs'
        # Return integer class indices (not one-hot) so CrossEntropyLoss can be used
        trainset = GTSRB(root_dir=dir, train=True, transform=transform_train,
                        target_transform=None)
        testset = GTSRB(root_dir=dir, train=False, transform=transform_val,
                        target_transform=None)
        

    elif args.dataset == 'data_Indian':
        print("load indian dataset")
        path = './datasets/signs/data_Indian'
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=(0.8), contrast=(1, 1)),
            transforms.Resize([config.DATA.IMG_SIZE, config.DATA.IMG_SIZE]),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

        ])
        
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Resize([config.DATA.IMG_SIZE, config.DATA.IMG_SIZE]),
        ])

            
        trainset = datasets.ImageFolder(path, transform=transform_train ,
                                             target_transform=None)
        testset = None

    elif args.dataset == 'data_china':

        print("load china dataset")
        
        path = './datasets/signs/data_china'
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=(0.8), contrast=(1, 1)),
            transforms.Resize([config.DATA.IMG_SIZE, config.DATA.IMG_SIZE]),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

        ])
        
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Resize([config.DATA.IMG_SIZE, config.DATA.IMG_SIZE]),
        ])

        trainset = datasets.ImageFolder(path, transform=transform_train,
                                             target_transform=None)
        testset = None

    else:
        print("key {} not found".format(args.dataset))

    if testset is None:

        train_size = int(0.7 * len(trainset))
        test_size = len(trainset) - train_size
        trainset, testset = random_split(trainset, [train_size, test_size])


    trainloader = DataLoader(trainset, batch_size=config.DATA.BATCH_SIZE, drop_last=True, shuffle=True)
    testloader = DataLoader(testset, batch_size=config.DATA.BATCH_SIZE, drop_last=True, shuffle=True)
    
    # setup mixup / cutmix 数据增强
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)
    

    return trainset, testset, trainloader, testloader, mixup_fn

