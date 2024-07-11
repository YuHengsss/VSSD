import os
from functools import partial
from typing import Callable

import torch
from torch import nn
from torch.utils import checkpoint

from mmengine.model import BaseModule
from mmdet.registry import MODELS as MODELS_MMDET
from mmseg.registry import MODELS as MODELS_MMSEG

def import_abspy(name="models", path="classification/"):
    import sys
    import importlib
    path = os.path.abspath(path)
    assert os.path.isdir(path)
    sys.path.insert(0, path)
    module = importlib.import_module(name)
    sys.path.pop(0)
    return module

build = import_abspy(
    "models", 
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../classification/"),
)
Backbone_VSSD: nn.Module = build.mamba2.Backbone_VMAMBA2

@MODELS_MMSEG.register_module()
@MODELS_MMDET.register_module()
class MM_VSSD(BaseModule, Backbone_VSSD):
    def __init__(self, *args, **kwargs):
        BaseModule.__init__(self)
        Backbone_VSSD.__init__(self, *args, **kwargs)

