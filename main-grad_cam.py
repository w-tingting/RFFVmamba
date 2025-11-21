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
import inspect
from pathlib import Path
from typing import Optional

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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision import transforms

from config import _C, get_config as build_config
from datasets import signs_datasets_singleGPU as datasets

from models import build_vssm_model
from timm.utils import ModelEma as ModelEma
from timm.utils import accuracy, AverageMeter
from utils.utils import NativeScalerWithGradNormCount, auto_resume_helper, reduce_tensor
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from utils.logger import create_logger

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from utils.optimizer import build_optimizer

warnings.filterwarnings("ignore")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT_DIR = os.path.abspath('/home/wtingting/Downloads/Mamba-code/RFFVMamba')
sys.path.append(ROOT_DIR)
from datasets.gtsrb_dataset import GTSRB
from torch.utils.data import DataLoader, random_split
from utils.lr_scheduler import build_scheduler


def log_model_complexity(model: torch.nn.Module, config, device: torch.device, logger: logging.Logger):
    """Compute and log trainable parameter count and FLOPs using fvcore."""
    params = parameter_count(model)[""] if callable(parameter_count) else sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    img_size = getattr(config.DATA, "IMG_SIZE", 224)
    channels = getattr(config.DATA, "IMG_CHANNELS", 3)
    dummy = torch.randn(1, channels, img_size, img_size, device=device)

    initial_mode = model.training
    model.eval()
    total_flops = None
    try:
        flops_analysis = FlopCountAnalysis(model, dummy)
        total_flops = flops_analysis.total()
        total_flops_gf = total_flops / 1e9
        logger.info(f"Trainable parameters: {params:,}")
        logger.info(f"Approximate FLOPs: {total_flops_gf:.6f} GFLOPs")
        logger.debug("FLOP breakdown:\n" + flop_count_str(flops_analysis))
    except Exception as exc:  # pragma: no cover - safeguard if fvcore fails
        logger.warning(f"Failed to compute FLOPs via fvcore: {exc}")
        logger.info(f"Trainable parameters: {params:,}")
    finally:
        if initial_mode:
            model.train()

    return params, total_flops


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _resolve_module(root: nn.Module, dotted_path: str) -> nn.Module:
    """Resolve a dotted attribute path on a module, raising a helpful error if missing."""
    module = root
    traversed = []
    for attr in dotted_path.split('.'):
        traversed.append(attr)
        if not hasattr(module, attr):
            path = '.'.join(traversed)
            raise AttributeError(f"Module '{root.__class__.__name__}' has no submodule '{path}'")
        module = getattr(module, attr)
    if not isinstance(module, nn.Module):
        raise AttributeError(f"Attribute '{dotted_path}' is not a torch.nn.Module")
    return module


def _denormalize_image(tensor: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    """Convert a normalized CHW tensor into an RGB numpy image in [0, 1]."""
    tensor = tensor.detach().cpu()
    mean_tensor = torch.tensor(mean).view(-1, 1, 1)
    std_tensor = torch.tensor(std).view(-1, 1, 1)
    image = tensor * std_tensor + mean_tensor
    image = image.clamp(0.0, 1.0)
    return image.permute(1, 2, 0).numpy()


def _extract_class_names(dataset) -> Optional[list]:
    """Try to recover class names from dataset or nested Subset wrappers."""
    if dataset is None:
        return None
    if hasattr(dataset, 'classes'):
        return list(dataset.classes)
    if hasattr(dataset, 'dataset'):
        return _extract_class_names(dataset.dataset)
    return None


def generate_gradcam(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: os.PathLike,
    logger: logging.Logger,
    layer_path: str = "patch_merge",
    max_samples: int = 4,
    mean: tuple = IMAGENET_MEAN,
    std: tuple = IMAGENET_STD,
    class_names: Optional[list] = None,
    tag: Optional[str] = None,
    images: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    save_inputs: bool = False,
    input_save_dir: Optional[os.PathLike] = None,
) -> None:
    """Generate Grad-CAM overlays for a few samples and save the plots."""
    if images is None or labels is None:
        if dataloader is None:
            logger.warning("Grad-CAM skipped: no dataloader provided.")
            return
        loader_iter = iter(dataloader)
        try:
            images, labels = next(loader_iter)
        except StopIteration:
            logger.warning("Grad-CAM skipped: dataloader is empty.")
            return

    try:
        target_layer = _resolve_module(model, layer_path)
    except AttributeError as exc:
        logger.error("Grad-CAM skipped: %s", exc)
        return

    images = images.to(device)
    labels = labels.to(device)
    sample_count = min(max_samples, images.size(0))
    use_cuda = device.type == "cuda"

    model_was_training = model.training
    model.eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = [ClassifierOutputTarget(int(labels[idx].item())) for idx in range(sample_count)]

    stem = (tag or layer_path.replace('.', '_'))
    tag_dir = output_dir / stem
    tag_dir.mkdir(parents=True, exist_ok=True)

    input_dir = None
    if save_inputs:
        if input_save_dir is not None:
            input_dir = Path(input_save_dir)
        else:
            input_dir = output_dir / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)

    cam_kwargs = {}
    gradcam_signature = inspect.signature(GradCAM.__init__)
    if "use_cuda" in gradcam_signature.parameters:
        cam_kwargs["use_cuda"] = use_cuda
    elif "device" in gradcam_signature.parameters:
        cam_kwargs["device"] = device

    with GradCAM(model=model, target_layers=[target_layer], **cam_kwargs) as cam:
        grayscale_cam = cam(input_tensor=images[:sample_count], targets=targets)

    for idx in range(sample_count):
        label_id = int(labels[idx].item())
        label_name = str(label_id)
        if class_names and 0 <= label_id < len(class_names):
            label_name = f"{label_id}-{class_names[label_id]}"

        rgb_image = _denormalize_image(images[idx], mean=mean, std=std).astype(np.float32)
        cam_array = np.array(grayscale_cam[idx], dtype=np.float32)
        cam_array = np.squeeze(cam_array)
        if cam_array.ndim == 0:
            cam_map = np.full(rgb_image.shape[:2], float(cam_array))
        elif cam_array.ndim == 1:
            cam_map = np.full(rgb_image.shape[:2], float(cam_array.mean()))
        elif cam_array.ndim == 2:
            cam_map = cam_array
        elif cam_array.ndim == 3:
            cam_map = cam_array.mean(axis=0)
        else:
            logger.warning("Skipping Grad-CAM for tag '%s' sample %d due to unsupported CAM shape %s", tag or layer_path, idx, cam_array.shape)
            continue

        cam_tensor = torch.tensor(cam_map, dtype=torch.float32)
        if cam_tensor.ndim == 2:
            cam_tensor = cam_tensor.unsqueeze(0).unsqueeze(0)
        elif cam_tensor.ndim == 3:
            cam_tensor = cam_tensor.unsqueeze(0)
        elif cam_tensor.ndim != 4:
            cam_tensor = cam_tensor.view(1, 1, 1, 1)

        resized = torch.nn.functional.interpolate(
            cam_tensor,
            size=rgb_image.shape[:2],
            mode="bilinear",
            align_corners=False,
        )
        cam_map = resized.squeeze().cpu().numpy()
        cam_map = cam_map - cam_map.min()
        if cam_map.max() > 0:
            cam_map = cam_map / cam_map.max()

        cam_overlay = show_cam_on_image(rgb_image, cam_map, use_rgb=True)

        if input_dir is not None:
            input_array = np.clip(rgb_image * 255.0, 0, 255).astype(np.uint8)
            input_path = input_dir / f"input_{idx}_label_{label_id}.png"
            if not input_path.exists():
                Image.fromarray(input_array).save(input_path)
                logger.info("Saved Grad-CAM input image to %s", input_path)

        overlay_path = tag_dir / f"gradcam_{stem}_{idx}_label_{label_id}.png"
        Image.fromarray(cam_overlay).save(overlay_path)
        logger.info("Saved Grad-CAM overlay for '%s' to %s", stem, overlay_path)

    if model_was_training:
        model.train()


def generate_gradcam_suite(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: os.PathLike,
    logger: logging.Logger,
    layer_map: dict,
    max_samples: int,
    mean: tuple,
    std: tuple,
    class_names: Optional[list],
) -> None:
    if dataloader is None:
        logger.warning("Grad-CAM suite skipped: no dataloader provided.")
        return

    loader_iter = iter(dataloader)
    try:
        images, labels = next(loader_iter)
    except StopIteration:
        logger.warning("Grad-CAM suite skipped: dataloader is empty.")
        return

    items = list(layer_map.items())
    input_dir = Path(output_dir) / "inputs"

    for idx, (tag, layer_path) in enumerate(items):
        generate_gradcam(
            model=model,
            dataloader=dataloader,
            device=device,
            output_dir=output_dir,
            logger=logger,
            layer_path=layer_path,
            max_samples=max_samples,
            mean=mean,
            std=std,
            class_names=class_names,
            tag=tag,
            images=images,
            labels=labels,
            save_inputs=(idx == 0),
            input_save_dir=input_dir,
        )


def evaluate_epoch(model, data_loader, device, criterion, config, logger, epoch=None):
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    all_preds = []
    all_targets = []

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.to(device)
        target = target.to(device)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)

        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        preds = output.argmax(dim=1)
        all_preds.append(preds.detach().cpu())
        all_targets.append(target.detach().cpu())

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
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

    return {
        "loss": loss_meter.avg,
        "acc1": acc1_meter.avg,
        "acc5": acc5_meter.avg,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "batch_time_avg": batch_time.avg,
    }


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
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
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

    parser.add_argument('--gradcam', type=str2bool, default=True,
                        help='Generate Grad-CAM visualizations for a batch of test images after training.')
    parser.add_argument('--gradcam-samples', type=int, default=4,
                        help='Number of images to visualize with Grad-CAM.')
    parser.add_argument('--gradcam-layer', type=str, default='patch_merge',
                        help='Dotted layer path to hook for Grad-CAM (default: patch_merge).')

    args, unparsed = parser.parse_known_args()
    config = build_config(args)

    return args, config


def train_and_test(train_dataloader, test_dataloader, model, device, mixup_fn, *, config, logger, tb_writer, model_ema, len_testset):
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(config, model, logger)

    loss_scaler = NativeScalerWithGradNormCount()
    steps_per_epoch = len(train_dataloader)
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, steps_per_epoch // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, steps_per_epoch)

    best_info = {
        "acc1": float("-inf"),
        "epoch": -1,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }
    best_checkpoint_path = os.path.join(config.OUTPUT, "best_model.pth")

    logger.info("Start training")
    start_time = time.time()

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        model.train()
        optimizer.zero_grad()

        num_steps = len(train_dataloader)

        batch_time = AverageMeter()
        model_time = AverageMeter()
        data_time = AverageMeter()
        loss_meter = AverageMeter()
        norm_meter = AverageMeter()
        scaler_meter = AverageMeter()

        epoch_start = time.time()
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

            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(
                loss,
                optimizer,
                clip_grad=config.TRAIN.CLIP_GRAD,
                parameters=model.parameters(),
                create_graph=is_second_order,
                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0,
            )

            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.zero_grad()
                lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
                if model_ema is not None:
                    model_ema.update(model)

            loss_scale_value = loss_scaler.state_dict()["scale"]

            loss_meter.update(loss.item(), targets.size(0))
            if grad_norm is not None:
                norm_meter.update(grad_norm)
            scaler_meter.update(loss_scale_value)
            batch_time.update(time.time() - end)
            end = time.time()

            model_time_warmup = config.TRAIN.WARMUP_EPOCHS
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

        epoch_time = time.time() - epoch_start
        logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
        if batch_time.count > 0:
            logger.info(f"Average batch train time: {batch_time.avg:.4f}s")

        tb_writer.add_scalar("train/loss", loss_meter.avg, epoch)
        tb_writer.add_scalar("train/grad_norm", norm_meter.avg, epoch)
        tb_writer.add_scalar("train/loss_scale", scaler_meter.avg, epoch)
        tb_writer.add_scalar("train/time_sec", epoch_time, epoch)
        tb_writer.add_scalar("train/time_per_batch_sec", batch_time.avg if batch_time.count else 0.0, epoch)
        tb_writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], epoch)

        val_metrics = evaluate_epoch(model, test_dataloader, device, criterion, config, logger, epoch)

        tb_writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        tb_writer.add_scalar("val/acc1", val_metrics["acc1"], epoch)
        tb_writer.add_scalar("val/acc5", val_metrics["acc5"], epoch)
        tb_writer.add_scalar("val/precision", val_metrics["precision"], epoch)
        tb_writer.add_scalar("val/recall", val_metrics["recall"], epoch)
        tb_writer.add_scalar("val/f1", val_metrics["f1"], epoch)

        if val_metrics["acc1"] > best_info["acc1"]:
            best_info.update({
                "acc1": val_metrics["acc1"],
                "epoch": epoch,
                "precision": val_metrics["precision"],
                "recall": val_metrics["recall"],
                "f1": val_metrics["f1"],
            })

            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "scaler": loss_scaler.state_dict(),
                "best_acc1": val_metrics["acc1"],
                "precision": val_metrics["precision"],
                "recall": val_metrics["recall"],
                "f1": val_metrics["f1"],
            }
            if model_ema is not None:
                checkpoint["model_ema"] = model_ema.ema.state_dict()

            torch.save(checkpoint, best_checkpoint_path)
            logger.info(
                f"New best model saved at epoch {epoch + 1} with Acc@1={val_metrics['acc1']:.3f}, "
                f"Precision={val_metrics['precision']:.4f}, Recall={val_metrics['recall']:.4f}, F1={val_metrics['f1']:.4f}"
            )

        if best_info["acc1"] > float("-inf"):
            logger.info(f"Best Acc@1 so far: {best_info['acc1']:.3f}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

    final_metrics = None
    if os.path.exists(best_checkpoint_path):
        best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(best_checkpoint["model"])
        logger.info(f"Loaded best model from epoch {best_checkpoint['epoch'] + 1}")
        final_metrics = evaluate_epoch(model, test_dataloader, device, criterion, config, logger)
        tb_writer.add_scalar("test/loss", final_metrics["loss"], config.TRAIN.EPOCHS)
        tb_writer.add_scalar("test/acc1", final_metrics["acc1"], config.TRAIN.EPOCHS)
        tb_writer.add_scalar("test/acc5", final_metrics["acc5"], config.TRAIN.EPOCHS)
        tb_writer.add_scalar("test/precision", final_metrics["precision"], config.TRAIN.EPOCHS)
        tb_writer.add_scalar("test/recall", final_metrics["recall"], config.TRAIN.EPOCHS)
        tb_writer.add_scalar("test/f1", final_metrics["f1"], config.TRAIN.EPOCHS)
        logger.info(f"Accuracy of the network on the {len_testset} test images: {final_metrics['acc1']:.3f}%")
        if best_info["acc1"] > float("-inf"):
            logger.info(f"Best Acc@1 so far: {best_info['acc1']:.3f}")
    else:
        logger.warning("Best checkpoint not found; skipping final evaluation with best weights.")

    return best_info, final_metrics, best_checkpoint_path
                    

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
    tb_writer = SummaryWriter(log_dir=os.path.join(config.OUTPUT, "tensorboard"))
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

    params, flops = log_model_complexity(model, config, device, logger)
    tb_writer.add_scalar("model/parameters", params, 0)
    if flops is not None:
        tb_writer.add_scalar("model/flops", flops / 1e9, 0)

    best_info, final_metrics, best_checkpoint_path = train_and_test(
        train_dataloader,
        test_dataloader,
        model,
        device,
        mixup_fn,
        config=config,
        logger=logger,
        tb_writer=tb_writer,
        model_ema=model_ema,
        len_testset=len(testset),
    )

    if args.gradcam:
        try:
            gradcam_dir = os.path.join(config.OUTPUT, "gradcam")
            class_names = _extract_class_names(testset)
            layer_targets = {
                "patch_embed": "patch_embed_cam",
                "fusion": "fusion_tap",
            }
            layer_targets.update({
                f"layer_{idx + 1}": f"layer_taps.{idx}" for idx in range(getattr(model, "num_layers", 0))
            })
            layer_targets["final_sum"] = "final_agg_tap"
            if args.gradcam_layer:
                custom_tag = f"custom_{args.gradcam_layer.replace('.', '_')}"
                layer_targets[custom_tag] = args.gradcam_layer

            generate_gradcam_suite(
                model=model,
                dataloader=test_dataloader,
                device=device,
                output_dir=gradcam_dir,
                logger=logger,
                layer_map=layer_targets,
                max_samples=args.gradcam_samples,
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
                class_names=class_names,
            )
        except Exception as exc:
            logger.exception("Grad-CAM generation failed: %s", exc)

    if best_info["epoch"] >= 0:
        logger.info(
            f"Best validation accuracy: {best_info['acc1']:.3f}% at epoch {best_info['epoch'] + 1} "
            f"(precision={best_info['precision']:.4f}, recall={best_info['recall']:.4f}, f1={best_info['f1']:.4f})"
        )
    else:
        logger.info("Best validation accuracy unavailable (no validation steps completed)")

    if final_metrics is not None:
        logger.info(
            f"Final test metrics — loss: {final_metrics['loss']:.4f}, acc1: {final_metrics['acc1']:.3f}, "
            f"acc5: {final_metrics['acc5']:.3f}, precision: {final_metrics['precision']:.4f}, "
            f"recall: {final_metrics['recall']:.4f}, f1: {final_metrics['f1']:.4f}"
        )

    tb_writer.close()

    log_file_path = os.path.join(config.OUTPUT, 'log.txt')
    if best_info["epoch"] >= 0 and os.path.exists(log_file_path):
        new_log_name = f"{args.dataset}_{best_info['acc1']:.4f}_log.txt"
        new_log_path = os.path.join(config.OUTPUT, new_log_name)

        for handler in logger.handlers[:]:
            handler.flush()
            handler.close()
            logger.removeHandler(handler)

        try:
            os.replace(log_file_path, new_log_path)
            print(f"Logs saved to {new_log_path}")
        except OSError:
            print(f"Failed to rename log file {log_file_path}")
        finally:
            logging.shutdown()
    else:
        logging.shutdown()
