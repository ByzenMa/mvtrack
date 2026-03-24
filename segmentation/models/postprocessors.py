# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
"""Postprocessors class to transform MDETR output according to the downstream task"""
from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import pycocotools.mask as mask_util

class PostProcessSegm(nn.Module):
    """Similar to PostProcess but for segmentation masks.
    This processor is to be called sequentially after PostProcess.
    Args:
        threshold: threshold that will be applied to binarize the segmentation masks.
    """

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, max_target_sizes):
        """Perform the computation
        Parameters:
            results: already pre-processed boxes (output of PostProcess) NOTE here
            outputs: raw outputs of the model
            orig_target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
            max_target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                              after data augmentation.
        """
        assert len(orig_target_sizes) == len(max_target_sizes)

        out_masks = outputs["pred_masks"]
        outputs_masks = out_masks.unsqueeze(1)
        outputs_masks = (outputs_masks.sigmoid() > self.threshold).cpu()
        for i, (cur_mask, t, tt) in enumerate(zip(outputs_masks, max_target_sizes, orig_target_sizes)):
            img_h, img_w = t[0], t[1]
            results.append({})
            results[i]["masks"] = cur_mask[:, :img_h, :img_w].unsqueeze(1)  # [:, 1, h, w]
            results[i]["masks"] = F.interpolate(
                results[i]["masks"].float(), size=tuple(tt.tolist()), mode="nearest"
            ).byte()
            results[i]["rle_masks"] = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in results[i]["masks"].cpu()]

        return results


def build_postprocessors():
    # for coco pretrain postprocessor
    postprocessors: Dict[str, nn.Module] = {"segm": PostProcessSegm(threshold=0.5)}
    return postprocessors
