import os
from functools import partial
import torch

from .vmamba import VSSM


def build_vssm_model(config, **kwargs):
    model = VSSM(
        patch_size=config.MODEL.VSSM.PATCH_SIZE, # 4
        in_chans=config.MODEL.VSSM.IN_CHANS, # 3
        num_classes=config.MODEL.NUM_CLASSES, # 1000
        depths=config.MODEL.VSSM.DEPTHS, # [2,2,9,2]
        dims=config.MODEL.VSSM.EMBED_DIM, # 96
        mc=config.MODEL.RFF.MC, # 5
        # ===================
        ssm_d_state=config.MODEL.VSSM.SSM_D_STATE, # 16
        ssm_ratio=config.MODEL.VSSM.SSM_RATIO, # 2.0
        ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO, # 2.0
        ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
        ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER, # silu
        ssm_conv=config.MODEL.VSSM.SSM_CONV, # 3
        ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
        ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE, # 0.0
        ssm_init=config.MODEL.VSSM.SSM_INIT, # v0
        forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE, # v2
        # ===================
        mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
        mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
        mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
        # ===================
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        patch_norm=config.MODEL.VSSM.PATCH_NORM,
        norm_layer=config.MODEL.VSSM.NORM_LAYER,
        downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
        patchembed_version=config.MODEL.VSSM.PATCHEMBED,
        gmlp=config.MODEL.VSSM.GMLP,
        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        # ===================
        posembed=config.MODEL.VSSM.POSEMBED,
        imgsize=config.DATA.IMG_SIZE,
        parallel_splits=tuple(config.MODEL.VSSM.PARALLEL_SPLITS),
    )
    return model






