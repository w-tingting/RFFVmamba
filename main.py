## -*- coding: utf-8 -*-
import argparse
import csv
from html import parser
import logging
import os
import random
import sys
import time
import warnings
import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import timm
import torch
import torch.nn as nn
import torch.nn.functional
import torchvision.models as models
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from thop import profile
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

from config import _C, get_config as build_config
from datasets import signs_datasets_singleGPU as datasets

from models import build_vssm_model
from timm.utils import ModelEma as ModelEma
from timm.utils import accuracy, AverageMeter
from utils.utils import  NativeScalerWithGradNormCount, auto_resume_helper, reduce_tensor
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from utils.logger import create_logger
from utils.utils import load_checkpoint_ema, load_pretrained_ema, save_checkpoint_ema

from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count
from utils.optimizer import build_optimizer

warnings.filterwarnings("ignore")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT_DIR = os.path.abspath('/home/wtingting/Downloads/Mamba-code/RFFVMamba')
sys.path.append(ROOT_DIR)
from datasets.gtsrb_dataset import GTSRB
from torch.utils.data import DataLoader, random_split
from utils.lr_scheduler import build_scheduler


def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def parse_option():
    parser = argparse.ArgumentParser(description='RFF Vmamba training and evaluation script')
    # 添加命令行参数
    # german 43
    # china 103
    # india 15
    parser.add_argument('--random-seed', type=int, default=42,
                        help='random seed(default: 42)')
    parser.add_argument('--dataset', type=str, default='GTSRB',
                        help='GTSRB CTSD BTSC data_Indian data_china(default: GTSRB)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--cfg', type=str, metavar="FILE", default="./configs/vssm/vmambav2v_tiny_224.yaml", help='path to config file', )
    parser.add_argument('--opts', default=None, nargs='+',
                        help='override config options as key value pairs')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
     # EMA related parameters
    parser.add_argument('--model_ema', type=str2bool, default=True)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')
    
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')

    args, unparsed = parser.parse_known_args()
    config = build_config(args)

    return args, config


def train_and_test(train_dataloader, test_dataloader, model, device, mixup_fn):
    
    # ****************************************************************************
    criterion = nn.CrossEntropyLoss() 
    optimizer = build_optimizer(config, model, logger)
   
    loss_scaler = NativeScalerWithGradNormCount()
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(train_dataloader) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(train_dataloader))
    # ****************************************************************************
    max_accuracy = 0.0
    max_accuracy_ema = 0.0
    logger.info("Start training")
    start_time = time.time()
    
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS): # TRAIN.START_EPOCH = 0 - TRAIN.EPOCHS = 300

        model.train()
        optimizer.zero_grad()
        
        num_steps = len(train_dataloader) #607

        batch_time = AverageMeter()
        model_time = AverageMeter()
        data_time = AverageMeter()
        loss_meter = AverageMeter()
        norm_meter = AverageMeter()
        scaler_meter = AverageMeter()
        
        start = time.time()
        end = time.time()
        for idx, (samples, targets) in enumerate(train_dataloader):
            samples = samples.to(device)
            targets = targets.to(device)

            if mixup_fn is not None:
                samples, targets = mixup_fn(samples, targets)

            data_time.update(time.time() - end)

            with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                outputs = model(samples)
            loss = criterion(outputs, targets)
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            # 梯度缩放，防止FP16下溢
            grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
            
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.zero_grad()
                lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]
            
            loss_meter.update(loss.item(), targets.size(0))
            if grad_norm is not None:  # loss_scaler return None if not update
                norm_meter.update(grad_norm)
            scaler_meter.update(loss_scale_value)
            batch_time.update(time.time() - end)
            end = time.time()
            
            model_time_warmup=config.TRAIN.WARMUP_EPOCHS
            if idx > model_time_warmup:
                model_time.update(batch_time.val - data_time.val)

            if idx % config.PRINT_FREQ == 0:
                lr = optimizer.param_groups[0]['lr']
                wd = optimizer.param_groups[0]['weight_decay']
                
                etas = batch_time.avg * (num_steps - idx)
                logger.info(
                    f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                    f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                    f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    f'data time {data_time.val:.4f} ({data_time.avg:.4f})\t'
                    f'model time {model_time.val:.4f} ({model_time.avg:.4f})\t'
                    f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                    f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})'
                    )
        epoch_time = time.time() - start
        logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
        
        
        if (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint_ema(config, epoch, model, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger, model_ema, max_accuracy_ema)
            
        # validation
        model.eval()

        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        acc1_meter = AverageMeter()
        acc5_meter = AverageMeter()
        all_preds = []
        all_targets = []
        end = time.time()
        for idx, (images, target) in enumerate(test_dataloader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                output = model(images)

            # measure accuracy and record loss
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            preds = output.argmax(dim=1)
            all_preds.append(preds.detach().cpu())
            all_targets.append(target.detach().cpu())


            loss_meter.update(loss.item(), target.size(0))
            acc1_meter.update(acc1.item(), target.size(0))
            acc5_meter.update(acc5.item(), target.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % config.PRINT_FREQ == 0:
                
                logger.info(
                    f'Test: [{idx}/{len(test_dataloader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                    f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})'
                    )
        if all_preds:
            y_true = torch.cat(all_targets).numpy()
            y_pred = torch.cat(all_preds).numpy()
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        else:
            precision = recall = f1 = 0.0

        logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
        logger.info(f'   Precision {precision:.4f} Recall {recall:.4f} F1 {f1:.4f}')
        
        logger.info(f"Accuracy of the network on the {len(testset)} test images: {acc1:.1f}%")
        max_accuracy = max(max_accuracy, acc1_meter.avg)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
                    

if __name__ == '__main__':
     # parse args
    args, config = parse_option()
    # ****************************************************************************
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:1" ) 
    # logger
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")
    # ****************************************************************************
    # Load Datasets
    trainset, testset, train_dataloader, test_dataloader, mixup_fn = datasets.init_dataset(config, args)
    # ****************************************************************************
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}") #'vssm'/'vssm_tiny_224'
    # # init model
    model = build_vssm_model(config)
    model = model.to(device)
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)
    # # ****************************************************************************
    if hasattr(model, 'flops'):
        logger.info(str(model))
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"number of params: {n_parameters}")
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")
    else:
        logger.info(flop_count_str(FlopCountAnalysis(model, (trainset[0][0][None],))))
        
    train_and_test(train_dataloader, test_dataloader, model, device, mixup_fn)
