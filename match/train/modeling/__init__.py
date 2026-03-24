# encoding: utf-8

from .baseline_modular import ConfigurableBaseline
from .baseline import Baseline


def build_model(cfg, num_classes):
    use_modular_model = any([
        cfg.MODEL.MODULES.USE_MSF,
        cfg.MODEL.MODULES.USE_RSCAMA,
        cfg.MODEL.MODULES.USE_TFF,
        cfg.MODEL.MODULES.USE_SFF,
    ])
    print(f"use msf module {cfg.MODEL.MODULES.USE_MSF}, use rscama module {cfg.MODEL.MODULES.USE_RSCAMA}, use tff module {cfg.MODEL.MODULES.USE_TFF}, use sff module {cfg.MODEL.MODULES.USE_SFF}")

    if use_modular_model:
        model = ConfigurableBaseline(
            num_classes,
            cfg.MODEL.LAST_STRIDE,
            cfg.MODEL.PRETRAIN_PATH,
            cfg.MODEL.NECK,
            cfg.TEST.NECK_FEAT,
            cfg.MODEL.NAME,
            cfg.MODEL.PRETRAIN_CHOICE,
            cfg,
        )
    else:
        model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK,
                         cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE)
    return model
