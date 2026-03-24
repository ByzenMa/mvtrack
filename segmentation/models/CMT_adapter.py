import torch
from torch import nn
import einops
from .layers import AdapterCA, HSA, PositionEmbeddingSine1D
from .sam2.utils.misc import trunc_normal_
from models.sam2.modeling.sam2_utils import LayerNorm2d

class CMT_adapter(nn.Module):

    def __init__(self,
                 in_channels_vis,
                 in_channels_txt,
                 adapter_channels,
                 view_token_dim,
                 HSA_patch_size=2,
                 args=None):
        super().__init__()
        self.adapter_channels = adapter_channels

        self.proj_vis_down = nn.Sequential(
            nn.Conv2d(in_channels_vis, adapter_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(adapter_channels), nn.ReLU(True))
        self.proj_vis_up = nn.Sequential(
            nn.ConvTranspose2d(adapter_channels, in_channels_vis, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels_vis), nn.ReLU(True))

        self.proj_text_down = nn.Sequential(nn.Linear(in_channels_txt, adapter_channels, bias=False)) # 512
        self.proj_text_up = nn.Sequential(nn.Linear(adapter_channels, in_channels_txt, bias=False))
        self.proj_view_down = nn.Linear(view_token_dim, adapter_channels, bias=False)
        self.proj_view_up = nn.Linear(adapter_channels, view_token_dim, bias=False)
        self.ca_V2T = AdapterCA(d_model=adapter_channels, nhead=8, dropout=0.0)
        self.ca_T2V = AdapterCA(d_model=adapter_channels, nhead=8, dropout=0.0)

        if args.HSA:
            self.hsa = HSA(d_model=adapter_channels, nhead=8)
            self.spatial_pos_embed = nn.Parameter(torch.zeros(1, 1, 1, HSA_patch_size*HSA_patch_size, adapter_channels))
            self.time_embed = PositionEmbeddingSine1D(adapter_channels, normalize=True)
            trunc_normal_(self.spatial_pos_embed, std=.02)
            self.norm_pre_HSA = nn.LayerNorm(adapter_channels)

        self.use_HSA = args.HSA
        self.HSA_patch_size = HSA_patch_size

    def forward(self, vis, clip_length, text, view_tokens=None):
        BT, C, H, W = vis.size()
        # proj down the 2 modalities
        x = self.proj_vis_down(vis)
        x = einops.rearrange(x, '(b t) c h w -> (h w) (b t) c', t=clip_length)
        t = self.proj_text_down(text)

        view_token_b = None
        if view_tokens is not None:
            # view_tokens as one extra text token in CMT
            view_token_b = self.proj_view_down(view_tokens)  # [B, Ca]
            t = torch.cat([t, view_token_b.unsqueeze(0)], dim=0)

        # Cross-Modal Temporal Attention
        x_atten = x
        if self.use_HSA:
            # hierarchical selective attention
            x_hsa = einops.rearrange(x, '(h w) (b t) c -> b t h w c', t=clip_length, h=H, w=W)
            x_hsa = einops.rearrange(x_hsa, 'b t (h1 h) (w1 w) c -> b t (h1 w1) (h w) c', h=self.HSA_patch_size, w=self.HSA_patch_size) # h1xw1 patches of size hxw
            x_hsa = x_hsa + self.time_embed(x_hsa).permute((0, 2, 1))[:, :, None, None, :] + self.spatial_pos_embed
            x_hsa = einops.rearrange(x_hsa, 'b t N p c -> (t p) (b N) c') # p = size of the patch = HSA_patch_size^2
            x_hsa = self.norm_pre_HSA(x_hsa)
            x_hsa = self.hsa(x_hsa)
            x_hsa = einops.rearrange(x_hsa,'(t p) (b N) c -> b t N p c', t=int(clip_length), b=int(BT/clip_length), p=self.HSA_patch_size**2)
            x_hsa = einops.rearrange(x_hsa, 'b t (h1 w1) (h w) c -> (h1 h w1 w) ( b t) c', h1=int(H/self.HSA_patch_size), w1=int(W/self.HSA_patch_size), h=self.HSA_patch_size, w=self.HSA_patch_size)
            x_atten = x_atten + x_hsa

        # cross modal adaptation
        x_atten = self.ca_V2T(x_atten, memory=t.repeat_interleave(clip_length, 1)) # HW, BT, Ca -> BT, Ca, H, W
        x_atten = einops.rearrange(x_atten, '(h w) (b t) c -> (b t) c h w', h=H, w=W, t=clip_length)
        t_atten = self.ca_T2V(t, einops.rearrange(x, 'hw (b t) c -> hw b t c', t=clip_length).mean(2))  # einops.rearrange(x, '(h w) (b t) c -> (h w) b t c', t=T)

        if view_token_b is not None:
            # split: text tokens + extra view token
            t_atten_text = t_atten[:-1]
            view_token_out_b = t_atten[-1]
        else:
            t_atten_text = t_atten
            view_token_out_b = None

        # proj up the 2 modalities
        x_out = self.proj_vis_up(x_atten)
        t_out = self.proj_text_up(t_atten_text)
        view_token_out = self.proj_view_up(view_token_out_b)

        return x_out, t_out, view_token_out
