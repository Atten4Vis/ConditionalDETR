# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch

from models.backbone import Backbone, Joiner
from models.conditional_detr import ConditionalDETR, PostProcess
from models.position_encoding import PositionEmbeddingSine
from models.segmentation import DETRsegm, PostProcessPanoptic
from models.transformer import Transformer

dependencies = ["torch", "torchvision"]


def _make_conditional_detr(backbone_name: str, dilation=False, num_classes=91, mask=False):
    hidden_dim = 256
    backbone = Backbone(backbone_name, train_backbone=True, return_interm_layers=mask, dilation=dilation)
    pos_enc = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
    backbone_with_pos_enc = Joiner(backbone, pos_enc)
    backbone_with_pos_enc.num_channels = backbone.num_channels
    transformer = Transformer(d_model=hidden_dim, return_intermediate_dec=True)
    detr = ConditionalDETR(backbone_with_pos_enc, transformer, num_classes=num_classes, num_queries=300)
    if mask:
        return DETRsegm(detr)
    return detr


def conditional_detr_resnet50(pretrained=False, num_classes=91, return_postprocessor=False):
    """
    ConditionalDETR R50 with 6 encoder and 6 decoder layers.

    Achieves 40.9 AP on COCO val5k.
    """
    model = _make_conditional_detr("resnet50", dilation=False, num_classes=num_classes)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://github.com/DeppMeng/ConditionalDETR/releases/download/v1.0/ConditionalDETR_r50_epoch50.pth", map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model


def conditional_detr_resnet50_dc5(pretrained=False, num_classes=91, return_postprocessor=False):
    """
    ConditionalDETR-DC5 R50 with 6 encoder and 6 decoder layers.

    The last block of RessNet-50 has dilation to increase
    output resolution.
    Achieves 43. AP on COCO val5k.
    """
    model = _make_conditional_detr("resnet50", dilation=True, num_classes=num_classes)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://github.com/DeppMeng/ConditionalDETR/releases/download/v1.0/ConditionalDETR_r50dc5_epoch50.pth", map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model


def conditional_detr_resnet101(pretrained=False, num_classes=91, return_postprocessor=False):
    """
    ConditionalDETR-DC5 R101 with 6 encoder and 6 decoder layers.

    Achieves 42.8 AP on COCO val5k.
    """
    model = _make_conditional_detr("resnet101", dilation=False, num_classes=num_classes)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://github.com/DeppMeng/ConditionalDETR/releases/download/v1.0/ConditionalDETR_r101_epoch50.pth", map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model


def conditional_detr_resnet101_dc5(pretrained=False, num_classes=91, return_postprocessor=False):
    """
    ConditionalDETR-DC5 R101 with 6 encoder and 6 decoder layers.

    The last block of ResNet-101 has dilation to increase
    output resolution.
    Achieves 45.0 AP on COCO val5k.
    """
    model = _make_conditional_detr("resnet101", dilation=True, num_classes=num_classes)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://github.com/DeppMeng/ConditionalDETR/releases/download/v1.0/ConditionalDETR_r101dc5_epoch50.pth", map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model

