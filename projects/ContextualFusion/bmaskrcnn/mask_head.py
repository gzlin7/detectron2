# Copyright (c) wondervictor. All Rights Reserved
from typing import List
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage

from detectron2.modeling.roi_heads import ROI_MASK_HEAD_REGISTRY
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_inference


def dice_loss_func(input, target):
    smooth = 1.
    n = input.size(0)
    iflat = input.view(n, -1)
    tflat = target.view(n, -1)
    intersection = (iflat * tflat).sum(1)
    loss = 1 - ((2. * intersection + smooth) /
                (iflat.sum(1) + tflat.sum(1) + smooth))
    return loss.mean()

@torch.jit.unused
def mask_rcnn_loss(pred_mask_logits: torch.Tensor, instances: List[Instances], vis_period: int = 0):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
        vis_period (int): the period (in steps) to dump visualization.

    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits.device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)

    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0

    gt_masks = cat(gt_masks, dim=0)

    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]

    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5
    gt_masks = gt_masks.to(dtype=torch.float32)

    # Log the training accuracy (using gt classes and 0.5 threshold)
    mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

    storage = get_event_storage()
    storage.put_scalar("mask_rcnn/accuracy", mask_accuracy)
    storage.put_scalar("mask_rcnn/false_positive", false_positive)
    storage.put_scalar("mask_rcnn/false_negative", false_negative)
    if vis_period > 0 and storage.iter % vis_period == 0:
        pred_masks = pred_mask_logits.sigmoid()
        vis_masks = torch.cat([pred_masks, gt_masks], axis=2)
        name = "Left: mask prediction;   Right: mask GT"
        for idx, vis_mask in enumerate(vis_masks):
            vis_mask = torch.stack([vis_mask] * 3, axis=0)
            storage.put_image(name + f" ({idx})", vis_mask)

    mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction="mean")
    return mask_loss


@ROI_MASK_HEAD_REGISTRY.register()
class ContextualFusionHead(nn.Module):

    def __init__(self, cfg, input_shape: ShapeSpec, vis_period: int = 0, loss_weight: float = 1.0):
        super(ContextualFusionHead, self).__init__()

        self.vis_period = vis_period
        self.loss_weight = loss_weight

        conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        conv_norm = cfg.MODEL.ROI_MASK_HEAD.NORM
        num_context_conv = cfg.MODEL.CONTEXT_MASK_HEAD.NUM_CONV
        num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
            num_classes = 1

        self.mask_fcns = []
        cur_channels = input_shape.channels
        for k in range(num_conv):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.mask_fcns.append(conv)
            cur_channels = conv_dim

        self.mask_final_fusion = Conv2d(
            conv_dim, conv_dim,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=not conv_norm,
            norm=get_norm(conv_norm, conv_dim),
            activation=F.relu)

        self.downsample = Conv2d(
            conv_dim, conv_dim,
            kernel_size=3,
            padding=1,
            stride=2,
            bias=not conv_norm,
            norm=get_norm(conv_norm, conv_dim),
            activation=F.relu
        )
        self.context_fcns = []
        cur_channels = input_shape.channels
        for k in range(num_context_conv):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("boundary_fcn{}".format(k + 1), conv)
            self.context_fcns.append(conv)
            cur_channels = conv_dim

        self.context_to_mask = Conv2d(
            conv_dim, conv_dim,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=not conv_norm,
            norm=get_norm(conv_norm, conv_dim),
            activation=F.relu
        )

        self.mask_deconv = ConvTranspose2d(
            conv_dim, conv_dim, kernel_size=2, stride=2, padding=0
        )
        self.mask_predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)

        for layer in self.mask_fcns + self.context_fcns +\
                     [self.mask_deconv, self.context_to_mask,
                      self.mask_final_fusion, self.downsample]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.mask_predictor.weight, std=0.001)
        if self.mask_predictor.bias is not None:
            nn.init.constant_(self.mask_predictor.bias, 0)

    def forward(self, mask_features, context_feature, instances: List[Instances]):
        # self.mask_fcns
        # self.downsample
        # self.context_fcns
        # self.context_to_mask
        # self.mask_final_fusion
        # self.mask_deconv
        # self.mask_predictor

        for layer in self.mask_fcns:
            mask_features = layer(mask_features)
        # downsample
        context_feature = self.downsample(context_feature)

        for layer in self.context_fcns:
            context_feature = layer(context_feature)
        # boundary to mask fusion
        mask_features = self.context_to_mask(context_feature) + mask_features
        mask_features = self.mask_final_fusion(mask_features)
        # mask prediction
        mask_features = F.relu(self.mask_deconv(mask_features))
        mask_logits = self.mask_predictor(mask_features)

        if self.training:
            return {"loss_mask": mask_rcnn_loss(mask_logits, instances, self.vis_period) * self.loss_weight}
        else:
            mask_rcnn_inference(mask_logits, instances)
            return instances