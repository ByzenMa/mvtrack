# encoding: utf-8
"""Configurable baseline variant that keeps the original baseline isolated."""

import importlib.util
from pathlib import Path
import torch

from torch import nn
import torch.nn.functional as F

from .baseline_base import BaselineBase, weights_init_classifier, weights_init_kaiming


class ConfigurableBaseline(BaselineBase):
    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, cfg):
        super(ConfigurableBaseline, self).__init__(
            num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice
        )
        modules_cfg = cfg.MODEL.MODULES
        self.use_msf = modules_cfg.USE_MSF
        self.use_rscama = modules_cfg.USE_RSCAMA
        self.use_tff = modules_cfg.USE_TFF
        self.use_sff = modules_cfg.USE_SFF

        self.module_order = [
            ('msf', self.use_msf),
            ('rscama', self.use_rscama),
            ('tff', self.use_tff),
            ('sff', self.use_sff),
        ]

        if self.use_msf:
            self.msf_low = nn.Conv1d(self.in_planes, self.in_planes, kernel_size=1, bias=False)
            self.msf_mid = nn.Conv1d(self.in_planes, self.in_planes, kernel_size=1, bias=False)
            self.msf_high = nn.Conv1d(self.in_planes, self.in_planes, kernel_size=1, bias=False)
            self.msf = self._load_msf()(self.in_planes)

        if self.use_rscama:
            self.rscama = self._build_rscama(modules_cfg)

        if self.use_tff:
            tff_module, _ = self._load_tff_sff()
            self.tff_branch_a = nn.Conv2d(self.in_planes, self.in_planes, kernel_size=1, bias=False)
            self.tff_branch_b = nn.Conv2d(self.in_planes, self.in_planes, kernel_size=1, bias=False)
            self.tff = tff_module(self.in_planes, self.in_planes)

        if self.use_sff:
            _, sff_module = self._load_tff_sff()
            self.sff = sff_module(self.in_planes)

        enhanced_num = sum([self.use_msf, self.use_rscama, self.use_tff, self.use_sff]) + 1
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(self.in_planes * enhanced_num, self.in_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.in_planes),
            nn.ReLU(inplace=True),
        )

        if self.neck == 'bnneck':
            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def _load_module_from_file(self, file_name, module_name):
        module_path = Path(__file__).resolve().parent.parent / 'layers' / file_name
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _load_msf(self):
        return self._load_module_from_file('MSF.py', 'traffic_brain_layers_msf').MSF

    def _load_tff_sff(self):
        module = self._load_module_from_file('TFF&SFF.py', 'traffic_brain_layers_tff_sff')
        return module.TFF, module.SFF

    def _build_rscama(self, modules_cfg):
        try:
            module = self._load_module_from_file('RSCaMa.py', 'traffic_brain_layers_rscama')
        except Exception as exc:
            raise ImportError('Failed to import RSCaMa.py dependencies. Please ensure transformers+mamba dependencies are installed.') from exc

        rscama_cfg = module.MambaConfig(
            hidden_size=self.in_planes,
            state_size=modules_cfg.RSCAMA_STATE_SIZE,
            intermediate_size=modules_cfg.RSCAMA_INTERMEDIATE_SIZE,
            conv_kernel=modules_cfg.RSCAMA_CONV_KERNEL,
            num_hidden_layers=modules_cfg.RSCAMA_NUM_LAYERS,
            use_bias=True,
            use_conv_bias=True,
        )
        return module.CaMambaModel(rscama_cfg)

    def _apply_msf(self, feat_map):
        b, c, h, w = feat_map.shape
        seq = feat_map.view(b, c, h * w)
        fused = self.msf(self.msf_low(seq), self.msf_mid(seq), self.msf_high(seq))
        return fused.view(b, c, h, w)

    def _apply_rscama(self, feat_map):
        b, c, h, w = feat_map.shape
        seq = feat_map.flatten(2).transpose(1, 2).contiguous()
        output = self.rscama(inputs_embeds=seq, inputs_embeds_2=seq)
        if hasattr(output, 'last_hidden_state'):
            output = output.last_hidden_state
        elif isinstance(output, tuple):
            output = output[0]
        return output.transpose(1, 2).contiguous().view(b, c, h, w)

    def _apply_tff(self, feat_map):
        branch_a = self.tff_branch_a(feat_map)
        branch_b = self.tff_branch_b(feat_map)
        return self.tff(branch_a, branch_b)

    def _apply_sff(self, feat_map):
        small = F.avg_pool2d(feat_map, kernel_size=2, stride=2, ceil_mode=True)
        return self.sff(small, feat_map)

    def _enhance_features(self, feat_map):
        enhanced = feat_map
        feats = []
        feats.append(feat_map)
        for name, enabled in self.module_order:
            if not enabled:
                continue
            if name == 'msf':
                enhanced = self._apply_msf(enhanced)
            elif name == 'rscama':
                enhanced = self._apply_rscama(enhanced)
            elif name == 'tff':
                enhanced = self._apply_tff(enhanced)
            elif name == 'sff':
                enhanced = self._apply_sff(enhanced)
            feats.append(enhanced)
            enhanced = feat_map
        return self.feature_fusion(torch.cat(feats, dim=1))

    def forward(self, x):
        base_feat = self.base(x)
        base_feat = self._enhance_features(base_feat)

        global_feat = self.gap(base_feat)
        global_feat = global_feat.view(global_feat.shape[0], -1)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat
        if self.neck_feat == 'after':
            return feat
        return global_feat