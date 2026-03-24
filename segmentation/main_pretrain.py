"""
Training script of SAMWISE
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import train_one_epoch, evaluate
from models.samwise import build_samwise
import sys
from os.path import join
from datasets.coco_eval import CocoEvaluator
from collections import namedtuple
from models.postprocessors import build_postprocessors
import opts


def main(args):
    print(args.__dict__)
    utils.init_distributed_mode(args)
    if args.output_dir and utils.get_rank() == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        args.log_file = join(args.output_dir, 'log.txt')
        with open(args.log_file, 'w') as fp:
            fp.writelines(" ".join(sys.argv) + '\n')
            fp.writelines(str(args.__dict__) + '\n\n')

    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = build_samwise(args)

    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    n_parameters_tot = sum(p.numel() for p in model.parameters())
    print('number of params:', n_parameters_tot)

    head = []
    fix = []
    for k, v in model_without_ddp.named_parameters():
        if v.requires_grad:
            head.append(v)
        else:
            fix.append(v)

    print("Trainable parameters: ", sum(p.numel() for p in head))
    print("Parameters fixed: ", sum(p.numel() for p in fix))

    param_list = [{
        'params': head,
        'initial_lr': args.lr
    }]

    optimizer = torch.optim.AdamW(param_list, lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drop)

    dataset_names = ["refcoco", "refcoco+", "refcocog"]
    dataset_train = torch.utils.data.ConcatDataset(
        [build_dataset(name, image_set="train", args=args) for name in dataset_names]
    )

    args.batch_size = int(args.batch_size/args.ngpu)
    if args.distributed:
        sampler_train = samplers.DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)

    Val_all = namedtuple(typename="val_data", field_names=["dataset_name", "dataloader", "base_ds", "evaluator_list"])
    dataset_names = ["refcoco"]

    val_tuples = []
    for name in dataset_names:
        dataset_val = build_dataset(name, image_set="val", args=args)
        sampler_val = (
            samplers.DistributedSampler(dataset_val, shuffle=False) if args.distributed else torch.utils.data.SequentialSampler(dataset_val)
        )
        data_loader_val = DataLoader(
            dataset_val,
            args.batch_size_val,
            sampler=sampler_val,
            drop_last=False,
            collate_fn=utils.collate_fn,
            num_workers=args.num_workers,
        )
        base_ds = get_coco_api_from_dataset(dataset_val)
        val_tuples.append(Val_all(dataset_name=name, dataloader=data_loader_val, base_ds=base_ds, evaluator_list=None))

    # build evaluator list for dataset_val
    def build_evaluator_list(base_ds):
        """Helper function to build the list of evaluators for a given dataset"""
        evaluator_list = []
        iou_types = []
        iou_types.append("segm")

        evaluator_list.append(CocoEvaluator(base_ds, tuple(iou_types), useCats=False))
        # TODO: currently ont support RefExpEvaluator (memory error)
        return evaluator_list

    output_dir = Path(args.output_dir)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        else:
            print("Model match")
        if not args.eval and args.resume_optimizer and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print(
                    'Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        print("Evaluating......")
        test_stats = {}
        for i, item in enumerate(val_tuples):
            evaluator_list = build_evaluator_list(item.base_ds)
            postprocessors = build_postprocessors()
            item = item._replace(evaluator_list=evaluator_list)
            print(f"Evaluating {item.dataset_name}")
            curr_test_stats = evaluate(
                model=model,
                postprocessors=postprocessors,
                data_loader=item.dataloader,
                device=device,
                args=args,
            )
            test_stats.update({item.dataset_name + "_" + k: v for k, v in curr_test_stats.items()})

        log_stats = {
            **{f"test_{k}": v for k, v in test_stats.items()},
            "n_parameters": n_parameters_tot,
        }
        print(log_stats)
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        return


    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
                model, data_loader_train, optimizer, device, epoch,
                args.clip_max_norm, lr_scheduler=lr_scheduler, args = args)

        if args.output_dir:
            print("Save Model")
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % 1 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)


        test_stats = {}
        for i, item in enumerate(val_tuples):
            evaluator_list = build_evaluator_list(item.base_ds)
            postprocessors = build_postprocessors()
            item = item._replace(evaluator_list=evaluator_list)
            print(f"Evaluating {item.dataset_name}")
            curr_test_stats = evaluate(
                model=model,
                postprocessors=postprocessors,
                data_loader=item.dataloader,
                device=device,
                args=args,
            )
            test_stats.update({item.dataset_name + "_" + k: v for k, v in curr_test_stats.items()})

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters_tot}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")



    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser('SAMWISE training and evaluation script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    name_exp = args.name_exp
    args.output_dir = os.path.join(args.output_dir, name_exp)

    main(args)




