_base_ = [
    '../swin/mask-rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py'
]

model = dict(
    backbone=dict(
        type='MM_VSSD',
        out_indices=(0, 1, 2, 3),
        pretrained="change to the path of the pretrained model",
        embed_dim=64,
        depths=(2, 4, 8, 4),
        num_heads=(2, 4, 8, 16),
        simple_downsample=False,
        simple_patch_embed=False,
        ssd_expand=2,
        ssd_chunk_size=256,
        linear_attn_duality=True,
        attn_types=['mamba2', 'mamba2', 'mamba2', 'standard'],
        bidirection=False,
        drop_path_rate=0.2,
        d_state=64,
        ssd_positve_dA=True,
    ),
    neck=dict(in_channels=[64, 128, 256, 512]),
)
