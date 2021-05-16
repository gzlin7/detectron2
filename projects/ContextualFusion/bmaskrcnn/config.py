# Copyright (c) wondervictor. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_context_preserving_config(cfg):

    cfg.MODEL.CONTEXT_MASK_HEAD = CN()
    cfg.MODEL.CONTEXT_MASK_HEAD.POOLER_RESOLUTION = 28
    cfg.MODEL.CONTEXT_MASK_HEAD.IN_FEATURES = ("p2",)
    cfg.MODEL.CONTEXT_MASK_HEAD.NUM_CONV = 2 

