_base_ = [
    '../swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py'
]

model = dict(
    backbone=dict(
        type='MM_VSSD',
        out_indices=(0, 1, 2, 3),
        pretrained="change to the path of the pretrained model",
        embed_dim=48,
        depths=(2, 2, 8, 4),
        num_heads=(2, 4, 8, 16),
        simple_downsample=False,
        simple_patch_embed=False,
        ssd_expand=2,
        ssd_chunk_size=256,
        linear_attn_duality=True,
        attn_types = ['mamba2', 'mamba2', 'mamba2', 'standard'],
        bidirection = False,
        drop_path_rate = 0.1,
        d_state = 48,
        ssd_positve_dA = True,
    ),
    neck=dict(in_channels=[48, 96, 192, 384]),
)
