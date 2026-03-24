"""
Train and eval functions used in main.py
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import math
import os
import sys
from typing import Iterable
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tools.metrics import calculate_precision_at_k_and_iou_metrics
import util.misc as utils
from torch.nn import functional as F
from models.segmentation import loss_masks


def _build_reid_batch_inputs(view_samples, view_targets, device):
    # flatten three views and use dataset-provided pid/tid/camid definitions
    imgs, pids, cam_ids, tids, timestamps = [], [], [], [], []

    batch_size = len(view_targets[0])

    for b in range(batch_size):
        for view_idx in range(len(view_samples)):
            video_tensor = view_samples[view_idx].tensors[b]  # [T, C, H, W]
            target = view_targets[view_idx][b]
            frame_ids = target["frames_idx"]
            frame_count = int(frame_ids.numel())

            pid = int(target["pid"].item()) if hasattr(target["pid"], "item") else int(target["pid"])
            tid = int(target["tid"].item()) if hasattr(target["tid"], "item") else int(target["tid"])
            cam_id = int(target["camid"].item()) if hasattr(target["camid"], "item") else int(target["camid"])

            for t in range(frame_count):
                imgs.append(video_tensor[t])
                pids.append(pid)
                cam_ids.append(cam_id)
                tids.append(tid)
                timestamps.append(int(frame_ids[t].item()) if hasattr(frame_ids[t], "item") else int(frame_ids[t]))

    x = torch.stack(imgs, dim=0).to(device)
    return {
        "images": x,
        "pids": torch.tensor(pids, dtype=torch.long, device=device),
        "cam_ids": torch.tensor(cam_ids, dtype=torch.long, device=device),
        "tids": torch.tensor(tids, dtype=torch.long, device=device),
        "timestamps": torch.tensor(timestamps, dtype=torch.float, device=device),
    }


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    lr_scheduler=None, args=None, reid_model=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    step=0

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        step+=1
        model.train()

        if args is not None and args.dataset_file == 'crtrack':
            losses = {}
            for view_idx in range(len(samples)):
                view_samples = samples[view_idx].to(device)
                view_targets = targets[view_idx]

                captions = [t["caption"] for t in view_targets]
                outputs = model(view_samples, captions, view_targets)

                view_loss_dict = loss_masks(
                    torch.cat(outputs["masks"]),
                    view_targets,
                    num_frames=view_samples.tensors.shape[1],
                )

                if args.use_cme_head and "pred_cme_logits" in outputs:
                    weight = torch.tensor([1., 2.]).to(device)
                    CME_loss = F.cross_entropy(
                        torch.cat(outputs["pred_cme_logits"]),
                        ignore_index=-1,
                        target=torch.tensor(outputs["cme_label"]).long().to(device),
                        weight=weight,
                    )
                    view_loss_dict["CME_loss"] = CME_loss if not CME_loss.isnan() else torch.tensor(0).to(device)

                for k, v in view_loss_dict.items():
                    if k in losses:
                        losses[k] = losses[k] + v
                    else:
                        losses[k] = v
            if reid_model is not None:
                reid_inputs = _build_reid_batch_inputs(samples, targets, device)
                reid_outputs = reid_model(
                    reid_inputs["images"],
                    reid_inputs["cam_ids"],
                    reid_inputs["timestamps"],
                )
                reid_loss_dict = reid_model.compute_loss(
                    outputs=reid_outputs,
                    pids=reid_inputs["pids"],
                    cam_ids=reid_inputs["cam_ids"],
                    tids=reid_inputs["tids"],
                    timestamps=reid_inputs["timestamps"],
                )
                losses.update({f"reid_{k}": v for k, v in reid_loss_dict.items() if k != "loss"})
                losses["samwise_loss"] = sum(losses[k] for k in list(losses.keys()) if not k.startswith("reid_"))
                losses["reid_loss"] = reid_loss_dict["loss"]
                losses["loss"] = args.samwise_loss_weight * losses["samwise_loss"] + args.reid_loss_weight * losses[
                    "reid_loss"]
        else:
            samples = samples.to(device)
            captions = [t["caption"] for t in targets]
            outputs = model(samples, captions, targets)
            losses = {}
            seg_loss = loss_masks(torch.cat(outputs["masks"]), targets, num_frames=samples.tensors.shape[1])
            losses.update(seg_loss)
            if args.use_cme_head and "pred_cme_logits" in outputs:
                weight = torch.tensor([1., 2.]).to(device)
                CME_loss = F.cross_entropy(torch.cat(outputs["pred_cme_logits"]), ignore_index=-1,
                                            target=torch.tensor(outputs["cme_label"]).long().to(device),
                                            weight=weight)
                losses.update({"CME_loss": CME_loss if not CME_loss.isnan() else torch.tensor(0).to(device)})

        loss_dict = losses
        losses = loss_dict.get("loss", sum(loss_dict[k] for k in loss_dict.keys()))
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v for k, v in loss_dict_reduced.items()}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()

        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)

        optimizer.step()
        lr_scheduler.step()


        # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_updates = dict(loss_dict_reduced_scaled)
        if "loss" not in metric_updates:
            metric_updates["loss"] = loss_value
        metric_logger.update(**metric_updates, **loss_dict_reduced_unscaled)

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, postprocessors, data_loader, device, args):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    predictions = []
    for samples, targets in metric_logger.log_every(data_loader, 20, header):
        dataset_name = targets[0]["dataset_name"]
        samples = samples.to(device)
        captions = [t["caption"] for t in targets]
        targets = utils.targets_to(targets, device)

        outputs = model(samples, captions, targets)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm']([], outputs, orig_target_sizes, target_sizes)

        # REC & RES predictions
        for p, target in zip(results, targets):
            for m in p['rle_masks']:
                predictions.append({'image_id': target['image_id'].item(),
                                    'category_id': 1,  # dummy label, as categories are not predicted in ref-vos
                                    'segmentation': m,
                                    'score': 1
                                    })

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # evaluate RES
    # gather and merge predictions from all gpus
    gathered_pred_lists = utils.all_gather(predictions)
    predictions = [p for p_list in gathered_pred_lists for p in p_list]

    eval_metrics = {}
    if utils.is_main_process():
        if dataset_name == 'refcoco':
            coco_gt = COCO(os.path.join(args.coco_path, 'refcoco/instances_refcoco_val.json'))
        elif dataset_name == 'refcoco+':
            coco_gt = COCO(os.path.join(args.coco_path, 'refcoco+/instances_refcoco+_val.json'))
        elif dataset_name == 'refcocog':
            coco_gt = COCO(os.path.join(args.coco_path, 'refcocog/instances_refcocog_val.json'))
        else:
            raise NotImplementedError
        coco_pred = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_pred, iouType='segm')
        coco_eval.params.useCats = 0  # ignore categories as they are not predicted in ref-vos task
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        ap_labels = ['mAP 0.5:0.95', 'AP 0.5', 'AP 0.75', 'AP 0.5:0.95 S', 'AP 0.5:0.95 M', 'AP 0.5:0.95 L']
        ap_metrics = coco_eval.stats[:6]
        eval_metrics = {l: m for l, m in zip(ap_labels, ap_metrics)}
        # Precision and IOU
        precision_at_k, overall_iou, mean_iou = calculate_precision_at_k_and_iou_metrics(coco_gt, coco_pred)
        eval_metrics.update({f'segm P@{k}': m for k, m in zip([0.5, 0.6, 0.7, 0.8, 0.9], precision_at_k)})
        eval_metrics.update({'segm overall_iou': overall_iou, 'segm mean_iou': mean_iou})
        print(eval_metrics)

    return eval_metrics
