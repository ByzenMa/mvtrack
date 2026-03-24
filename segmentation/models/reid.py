from typing import Callable, Dict, Optional

import torch
from torch import nn
from torchvision import models as tv_models


class TorchvisionResNetBackbone(nn.Module):
    """Feature extractor wrapper for torchvision ResNet models."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def normalize(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    return 1.0 * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)


def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    return dist.clamp(min=1e-12).sqrt()


def hard_example_mining(dist_mat: torch.Tensor, labels: torch.Tensor):
    n = dist_mat.size(0)
    is_pos = labels.expand(n, n).eq(labels.expand(n, n).t())
    is_neg = labels.expand(n, n).ne(labels.expand(n, n).t())

    dist_ap = torch.mul(dist_mat, is_pos.float())
    dist_ap, _ = torch.max(dist_ap, 1, keepdim=True)

    dist_an = torch.mul(dist_mat, is_neg.float())
    dist_an[dist_an == 0.0] = 100000000.0
    dist_an, _ = torch.min(dist_an, 1, keepdim=True)

    return dist_ap.squeeze(1), dist_an.squeeze(1)


class TripletLoss(object):
    def __init__(self, margin: Optional[float] = None):
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin) if margin is not None else nn.SoftMarginLoss()

    def __call__(self, global_feat: torch.Tensor, labels: torch.Tensor, normalize_feature: bool = False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes: int, epsilon: float = 0.1, use_gpu: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = self.logsoftmax(inputs)
        onehot = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu:
            onehot = onehot.cuda()
        onehot = (1 - self.epsilon) * onehot + self.epsilon / self.num_classes
        return (-onehot * log_probs).mean(0).sum()


class ClusterLoss(nn.Module):
    def __init__(
        self,
        margin: float = 10.0,
        use_gpu: bool = True,
        ordered: bool = True,
        ids_per_batch: int = 16,
        imgs_per_id: int = 4,
    ):
        super().__init__()
        self.use_gpu = use_gpu
        self.margin = margin
        self.ordered = ordered
        self.ids_per_batch = ids_per_batch
        self.imgs_per_id = imgs_per_id

    def _cluster_loss(self, features: torch.Tensor, targets: torch.Tensor):
        if self.use_gpu:
            if self.ordered and targets.size(0) == self.ids_per_batch * self.imgs_per_id:
                unique_labels = targets[0:targets.size(0):self.imgs_per_id]
            else:
                unique_labels = targets.cpu().unique().cuda()
        else:
            if self.ordered and targets.size(0) == self.ids_per_batch * self.imgs_per_id:
                unique_labels = targets[0:targets.size(0):self.imgs_per_id]
            else:
                unique_labels = targets.unique()

        inter_min_distance = torch.zeros(unique_labels.size(0), device=features.device)
        intra_max_distance = torch.zeros(unique_labels.size(0), device=features.device)
        center_features = torch.zeros(unique_labels.size(0), features.size(1), device=features.device)

        index = torch.arange(0, unique_labels.size(0), device=features.device)
        for i in range(unique_labels.size(0)):
            label = unique_labels[i]
            same_class_features = features[targets == label]
            center_features[i] = same_class_features.mean(dim=0)
            intra_class_distance = euclidean_dist(center_features[index == i], same_class_features)
            intra_max_distance[i] = intra_class_distance.max()

        for i in range(unique_labels.size(0)):
            inter_class_distance = euclidean_dist(center_features[index == i], center_features[index != i])
            inter_min_distance[i] = inter_class_distance.min()

        cluster_loss = torch.mean(torch.relu(intra_max_distance - inter_min_distance + self.margin))
        return cluster_loss, intra_max_distance, inter_min_distance

    def forward(self, features: torch.Tensor, targets: torch.Tensor):
        assert features.size(0) == targets.size(0), "features.size(0) must equal targets.size(0)"
        return self._cluster_loss(features, targets)


def weights_init_kaiming(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1 and getattr(m, "affine", False):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class ReID(nn.Module):
    """Standalone ReID algorithm class with visual + spatial-temporal cues + configurable losses."""

    def __init__(
        self,
        num_classes: int,
        num_cameras: int,
        model_name: str = "resnet50",
        last_stride: int = 1,
        feat_dim: int = 2048,
        st_dim: int = 128,
        # config style options similar to make_loss
        sampler: str = "softmax_triplet",
        metric_loss_type: str = "triplet",
        label_smooth: bool = True,
        # triplet / track-triplet
        triplet_margin: float = 0.3,
        track_triplet_margin: float = 0.3,
        track_weight: float = 1.0,
        # cluster
        cluster_margin: float = 10.0,
        ids_per_batch: int = 16,
        imgs_per_id: int = 4,
        # extra losses in this figure3 model
        w_cam: float = 0.2,
        w_st: float = 0.1,
        st_tau: float = 300.0,
    ):
        super().__init__()
        self.sampler = sampler
        self.metric_loss_type = metric_loss_type
        self.track_weight = track_weight
        self.w_cam = w_cam
        self.w_st = w_st
        self.st_tau = st_tau

        self.base, in_planes = self._build_backbone(model_name=model_name, last_stride=last_stride)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.cam_embed = nn.Embedding(num_cameras, st_dim)
        self.time_mlp = nn.Sequential(nn.Linear(1, st_dim), nn.ReLU(inplace=True), nn.Linear(st_dim, st_dim))
        self.st_fuse = nn.Sequential(nn.Linear(in_planes + st_dim, feat_dim), nn.BatchNorm1d(feat_dim), nn.ReLU(inplace=True))

        self.bnneck = nn.BatchNorm1d(feat_dim)
        self.bnneck.bias.requires_grad_(False)
        self.id_classifier = nn.Linear(feat_dim, num_classes, bias=False)
        self.cam_classifier = nn.Linear(feat_dim, num_cameras, bias=False)

        self.st_fuse.apply(weights_init_kaiming)
        self.bnneck.apply(weights_init_kaiming)
        self.id_classifier.apply(weights_init_classifier)
        self.cam_classifier.apply(weights_init_classifier)

        self.triplet = TripletLoss(margin=triplet_margin)
        self.track_triplet = TripletLoss(margin=track_triplet_margin)
        self.cluster = ClusterLoss(
            margin=cluster_margin,
            use_gpu=torch.cuda.is_available(),
            ordered=True,
            ids_per_batch=ids_per_batch,
            imgs_per_id=imgs_per_id,
        )

        self.id_loss = (
            CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=0.1, use_gpu=torch.cuda.is_available())
            if label_smooth
            else nn.CrossEntropyLoss()
        )
        self.cam_loss = nn.CrossEntropyLoss()

        self.loss_fn = self._build_loss_fn()

    @staticmethod
    def _build_backbone(model_name: str, last_stride: int):
        if model_name == "resnet18":
            model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
            in_planes = 512
        elif model_name == "resnet34":
            model = tv_models.resnet34(weights=tv_models.ResNet34_Weights.IMAGENET1K_V1)
            in_planes = 512
        elif model_name == "resnet50":
            model = tv_models.resnet50(pretrained=True)
            in_planes = 2048
        elif model_name == "resnet101":
            model = tv_models.resnet101(pretrained=True)
            in_planes = 2048
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        # keep compatibility with the previous last_stride setting
        if last_stride == 1:
            model.layer4[0].conv2.stride = (1, 1)
            if model.layer4[0].downsample is not None:
                model.layer4[0].downsample[0].stride = (1, 1)

        return TorchvisionResNetBackbone(model), in_planes

    def forward(self, x: torch.Tensor, cam_ids: torch.Tensor, timestamps: torch.Tensor) -> Dict[str, torch.Tensor]:
        base_feat = self.base(x)
        visual_feat = self.gap(base_feat).view(base_feat.shape[0], -1)

        cam_feat = self.cam_embed(cam_ids)
        t = timestamps.float().view(-1, 1)
        t = (t - t.mean()) / (t.std() + 1e-12)
        time_feat = self.time_mlp(t)

        st_feat = cam_feat + time_feat
        fused_feat = self.st_fuse(torch.cat([visual_feat, st_feat], dim=1))
        feat_bn = self.bnneck(fused_feat)

        return {
            "feat": fused_feat,
            "feat_bn": feat_bn,
            "id_logits": self.id_classifier(feat_bn),
            "cam_logits": self.cam_classifier(feat_bn),
        }

    def spatial_temporal_consistency_loss(
        self, feat: torch.Tensor, pids: torch.Tensor, cam_ids: torch.Tensor, timestamps: torch.Tensor
    ) -> torch.Tensor:
        dist = torch.cdist(feat, feat, p=2)
        same_id = pids[:, None].eq(pids[None, :])
        diff_cam = ~cam_ids[:, None].eq(cam_ids[None, :])
        pos_mask = same_id & diff_cam

        if pos_mask.sum() == 0:
            return feat.new_tensor(0.0)

        dt = (timestamps[:, None] - timestamps[None, :]).abs().float()
        time_weight = torch.exp(-dt / self.st_tau)
        return (dist * time_weight)[pos_mask].mean()

    def _metric_loss(self, feat: torch.Tensor, pids: torch.Tensor, tids: torch.Tensor) -> Dict[str, torch.Tensor]:
        metric = {}
        if self.metric_loss_type in ["triplet", "triplet_cluster"]:
            metric["loss_triplet"] = self.triplet(feat, pids)[0]
            # keep original style: triplet(feat, target) + track_weight * track_triplet(feat, tid)
            metric["loss_track_triplet"] = self.track_triplet(feat, tids)[0]
        if self.metric_loss_type in ["cluster", "triplet_cluster"]:
            metric["loss_cluster"] = self.cluster(feat, pids)[0]
        return metric

    def _build_loss_fn(self) -> Callable:
        """Build a configurable loss_fn in the same spirit as original make_loss."""

        def loss_fn(
            score: torch.Tensor,
            feat: torch.Tensor,
            target: torch.Tensor,
            tid: torch.Tensor,
            cam_score: Optional[torch.Tensor] = None,
            cam_target: Optional[torch.Tensor] = None,
            st_loss: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            losses: Dict[str, torch.Tensor] = {}

            # Classification branch by sampler style
            if self.sampler in ["softmax", "softmax_triplet"]:
                losses["loss_id"] = self.id_loss(score, target)
            elif self.sampler == "triplet":
                losses["loss_id"] = feat.new_tensor(0.0)
            else:
                raise ValueError(f"expected sampler in [softmax, triplet, softmax_triplet], got {self.sampler}")

            # Metric branch by configurable metric_loss_type
            metric_losses = self._metric_loss(feat, target, tid)
            losses.update(metric_losses)

            # Keep camera/st terms for Figure3 extension
            if cam_score is not None and cam_target is not None:
                losses["loss_cam"] = self.cam_loss(cam_score, cam_target)
            if st_loss is not None:
                losses["loss_st"] = st_loss

            # aggregate using original behavior for triplet-track + optional cluster
            total = losses["loss_id"]
            if "loss_triplet" in losses:
                total = total + losses["loss_triplet"] + self.track_weight * losses["loss_track_triplet"]
            if "loss_cluster" in losses:
                total = total + losses["loss_cluster"]
            if "loss_cam" in losses:
                total = total + self.w_cam * losses["loss_cam"]
            if "loss_st" in losses:
                total = total + self.w_st * losses["loss_st"]
            losses["loss"] = total
            return losses

        return loss_fn

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        pids: torch.Tensor,
        cam_ids: torch.Tensor,
        tids: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if timestamps is None:
            timestamps = torch.zeros_like(cam_ids, dtype=torch.float)

        st_loss = self.spatial_temporal_consistency_loss(outputs["feat"], pids, cam_ids, timestamps)
        return self.loss_fn(
            score=outputs["id_logits"],
            feat=outputs["feat"],
            target=pids,
            tid=tids,
            cam_score=outputs["cam_logits"],
            cam_target=cam_ids,
            st_loss=st_loss,
        )