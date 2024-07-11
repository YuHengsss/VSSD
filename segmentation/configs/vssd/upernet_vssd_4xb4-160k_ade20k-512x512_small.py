
_base_ = [
    '../swin/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py'
]
model = dict(
    backbone=dict(
        type='MM_VSSD',
        out_indices=(0, 1, 2, 3),
        pretrained="change to the path of the pretrained model",
        embed_dim=64,
        depths=(3, 4, 21, 5),
        num_heads=(2, 4, 8, 16),
        simple_downsample=False,
        simple_patch_embed=False,
        ssd_expand=2,
        ssd_chunk_size=256,
        linear_attn_duality=True,
        attn_types=['mamba2', 'mamba2', 'mamba2', 'standard'],
        bidirection=False,
        drop_path_rate=0.4,
        d_state=64,
        ssd_positve_dA=True,
        key='model_ema'
    ),
    decode_head=dict(in_channels=[64, 128, 256, 512], num_classes=150,act_cfg=dict(type='ReLU', inplace=False)),
    auxiliary_head=dict(in_channels=256, num_classes=150,act_cfg=dict(type='ReLU', inplace=False),)
)

# train_dataloader = dict(batch_size=4) # as gpus=4

