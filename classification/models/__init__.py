import os
from functools import partial

import torch
from .mamba2 import VMAMBA2

# still on developing...
def build_vssd_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE
    if model_type in ["vmamba2"]:
        model = VMAMBA2(
            image_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.VMAMBA2.PATCH_SIZE,
            in_chans=config.MODEL.VMAMBA2.IN_CHANS,
            embed_dim=config.MODEL.VMAMBA2.EMBED_DIM,
            depths=config.MODEL.VMAMBA2.DEPTHS,
            num_heads=config.MODEL.VMAMBA2.NUM_HEADS,
            mlp_ratio=config.MODEL.VMAMBA2.MLP_RATIO,
            drop_rate=config.MODEL.VMAMBA2.DROP_RATE,
            drop_path_rate=config.MODEL.VMAMBA2.DROP_PATH_RATE,
            simple_downsample=config.MODEL.VMAMBA2.SIMPLE_DOWNSAMPLE,
            simple_patch_embed=config.MODEL.VMAMBA2.SIMPLE_PATCH_EMBED,
            ssd_expansion=config.MODEL.VMAMBA2.SSD_EXPANSION,
            ssd_ngroups=config.MODEL.VMAMBA2.SSD_NGROUPS,
            ssd_chunk_size=config.MODEL.VMAMBA2.SSD_CHUNK_SIZE,
            linear_attn_duality = config.MODEL.VMAMBA2.LINEAR_ATTN_DUALITY,
            lepe=config.MODEL.VMAMBA2.LEPE,
            attn_types=config.MODEL.VMAMBA2.ATTN_TYPES,
            bidirection=config.MODEL.VMAMBA2.BIDIRECTION,
            d_state=config.MODEL.VMAMBA2.D_STATE,
            ssd_positve_dA = config.MODEL.VMAMBA2.SSD_POSITIVE_DA,
        )
        return model
    return None



# used for analyze
def build_mmpretrain_models(cfg="swin_tiny", ckpt=True, only_backbone=False, with_norm=True, **kwargs):
    import os
    from functools import partial
    from mmengine.runner import CheckpointLoader
    from mmpretrain.models import build_classifier, ImageClassifier, ConvNeXt, VisionTransformer, SwinTransformer
    from mmengine.config import Config
    config_root = os.path.join(os.path.dirname(__file__), "../../analyze/mmpretrain_configs/configs/") 
    
    CFGS = dict(
        swin_tiny=dict(
            model=Config.fromfile(os.path.join(config_root, "./swin_transformer/swin-tiny_16xb64_in1k.py")).to_dict()['model'], 
            ckpt="https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth",
        ),
        convnext_tiny=dict(
            model=Config.fromfile(os.path.join(config_root, "./convnext/convnext-tiny_32xb128_in1k.py")).to_dict()['model'], 
            ckpt="https://download.openmmlab.com/mmclassification/v0/convnext/convnext-tiny_32xb128_in1k_20221207-998cf3e9.pth",
        ),
        deit_small=dict(
            model=Config.fromfile(os.path.join(config_root, "./deit/deit-small_4xb256_in1k.py")).to_dict()['model'], 
            ckpt="https://download.openmmlab.com/mmclassification/v0/deit/deit-small_pt-4xb256_in1k_20220218-9425b9bb.pth",
        ),
        resnet50=dict(
            model=Config.fromfile(os.path.join(config_root, "./resnet/resnet50_8xb32_in1k.py")).to_dict()['model'], 
            ckpt="https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth",
        ),
        # ================================
        swin_small=dict(
            model=Config.fromfile(os.path.join(config_root, "./swin_transformer/swin-small_16xb64_in1k.py")).to_dict()['model'], 
            ckpt="https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_small_224_b16x64_300e_imagenet_20210615_110219-7f9d988b.pth",
        ),
        convnext_small=dict(
            model=Config.fromfile(os.path.join(config_root, "./convnext/convnext-small_32xb128_in1k.py")).to_dict()['model'], 
            ckpt="https://download.openmmlab.com/mmclassification/v0/convnext/convnext-small_32xb128_in1k_20221207-4ab7052c.pth",
        ),
        deit_base=dict(
            model=Config.fromfile(os.path.join(config_root, "./deit/deit-base_16xb64_in1k.py")).to_dict()['model'], 
            ckpt="https://download.openmmlab.com/mmclassification/v0/deit/deit-base_pt-16xb64_in1k_20220216-db63c16c.pth",
        ),
        resnet101=dict(
            model=Config.fromfile(os.path.join(config_root, "./resnet/resnet101_8xb32_in1k.py")).to_dict()['model'], 
            ckpt="https://download.openmmlab.com/mmclassification/v0/resnet/resnet101_8xb32_in1k_20210831-539c63f8.pth",
        ),
        # ================================
        swin_base=dict(
            model=Config.fromfile(os.path.join(config_root, "./swin_transformer/swin-base_16xb64_in1k.py")).to_dict()['model'], 
            ckpt="https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_base_224_b16x64_300e_imagenet_20210616_190742-93230b0d.pth",
        ),
        convnext_base=dict(
            model=Config.fromfile(os.path.join(config_root, "./convnext/convnext-base_32xb128_in1k.py")).to_dict()['model'], 
            ckpt="https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_32xb128_in1k_20221207-fbdb5eb9.pth",
        ),
        replknet_base=dict(
            # comment this "from mmpretrain.models import build_classifier" in __base__/models/replknet...
            model=Config.fromfile(os.path.join(config_root, "./replknet/replknet-31B_32xb64_in1k.py")).to_dict()['model'], 
            ckpt="https://download.openmmlab.com/mmclassification/v0/replknet/replknet-31B_3rdparty_in1k_20221118-fd08e268.pth",
        ),
    )

    if cfg not in CFGS:
        return None

    model: ImageClassifier = build_classifier(CFGS[cfg]['model'])
    if ckpt:
        model.load_state_dict(CheckpointLoader.load_checkpoint(CFGS[cfg]['ckpt'])['state_dict'])

    if only_backbone:
        if isinstance(model.backbone, ConvNeXt):
            model.backbone.gap_before_final_norm = False
        if isinstance(model.backbone, VisionTransformer):
            model.backbone.out_type = 'featmap'

        def forward_backbone(self: ImageClassifier, x):
            x = self.backbone(x)[-1]
            return x
        if not with_norm:
            setattr(model, f"norm{model.backbone.out_indices[-1]}", lambda x: x)
        model.forward = partial(forward_backbone, model)

    return model


def build_model(config, is_pretrain=False):
    model = None
    
    if model is None:
        model = build_vssd_model(config, is_pretrain)
    return model




