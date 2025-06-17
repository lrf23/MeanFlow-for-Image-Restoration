"""修改 z= (1 - t_) * x +t_*c#z是流路径"""
import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp
from timm.models.vision_transformer import Attention
import torch.nn.functional as F
from einops import repeat, pack, unpack,rearrange
from torch.cuda.amp import autocast
from functools import partial
import math
import os
import random
from functools import partial
from pathlib import Path
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from ema_pytorch import EMA
from PIL import Image
import time
from torch import einsum, nn
from torch.optim import Adam, RAdam
from torch.utils.data import DataLoader
from torchvision import utils
from tqdm.auto import tqdm
from src.UnetRes_Meanflow import tensor2img,metric_module,UnetRes
def modulate(x, scale, shift):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def set_seed(SEED):
    # initialize random seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
class TimestepEmbedder(nn.Module):
    def __init__(self, dim, nfreq=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(nfreq, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.nfreq = nfreq

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half_dim, dtype=torch.float32)
            / half_dim
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.nfreq)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes + 1, dim)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding(labels)
        return embeddings


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.g


class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True, qk_norm=True, norm_layer=RMSNorm)
        # flasth attn can not be used with jvp
        self.attn.fused_attn = False
        self.norm2 = RMSNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_dim, act_layer=approx_gelu, drop=0
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), scale_msa, shift_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), scale_mlp, shift_mlp)
        )
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim, patch_size, out_dim):
        super().__init__()
        self.norm_final = RMSNorm(dim)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class FeatureFusionConditioner(nn.Module):
    """通过特征融合处理图像条件"""
    def __init__(self, input_size, patch_size, in_channels, dim):
        super().__init__()
        self.patch_embed = PatchEmbed(input_size, patch_size, in_channels, dim)
        
        # 特征融合网络
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
        )
        
        # 全局特征提取
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
    def forward(self, degraded_images):
        # degraded_images: (B, C, H, W)
        B, C, H, W = degraded_images.shape
        
        # Patch embedding
        x = self.patch_embed(degraded_images)  # (B, num_patches, dim)
        
        # 重新整理为空间特征图
        patch_size = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(B, h, w, -1).permute(0, 3, 1, 2)  # (B, dim, h, w)
        
        # 特征融合
        fused_features = self.fusion_conv(x)  # (B, dim, h, w)
        
        # 全局特征
        global_feat = self.global_pool(fused_features).squeeze(-1).squeeze(-1)  # (B, dim)
        global_feat = self.global_proj(global_feat)  # (B, dim)
        
        return global_feat, fused_features
class MFDiT(nn.Module):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        dim=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        use_image_condition=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, dim)
        self.t_embedder = TimestepEmbedder(dim)
        self.r_embedder = TimestepEmbedder(dim)
        self.use_image_condition = use_image_condition
        if use_image_condition:
            self.cond_processor = FeatureFusionConditioner(input_size, patch_size, in_channels, dim)
        else:
            self.cond_processor = None

        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim), requires_grad=True)

        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(dim, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize cond embedding table:
        if self.cond_processor is not None:
            # 初始化 patch embedding
            w = self.cond_processor.patch_embed.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.cond_processor.patch_embed.proj.bias, 0)
            
            # 初始化特征融合卷积层
            for layer in self.cond_processor.fusion_conv:
                if isinstance(layer, nn.Conv2d):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
            
            # 初始化全局投影层
            for layer in self.cond_processor.global_proj:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
            

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, r, y=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        H, W = x.shape[-2:]

        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2

        t = self.t_embedder(t)                   # (N, D)
        r = self.r_embedder(r)
        # t = torch.cat([t, r], dim=-1)
        t = t + r

        # condition
        c = t
        if self.use_image_condition and y is not None:
            global_cond, spatial_cond = self.cond_processor(y)
            c = c + global_cond  # 添加全                            # (N, D)

        for i, block in enumerate(self.blocks):
            x = block(x, c)                      # (N, T, D)

        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x


# Positional embedding from:
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb



class Normalizer:
    # minmax for raw image, mean_std for vae latent
    def __init__(self, mode='minmax', mean=None, std=None):
        assert mode in ['minmax', 'mean_std'], "mode must be 'minmax' or 'mean_std'"
        self.mode = mode

        if mode == 'mean_std':
            if mean is None or std is None:
                raise ValueError("mean and std must be provided for 'mean_std' mode")
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

    @classmethod
    def from_list(cls, config):
        """
        config: [mode, mean, std]
        """
        mode, mean, std = config
        return cls(mode, mean, std)

    def norm(self, x):
        if self.mode == 'minmax':
            return x * 2 - 1
        elif self.mode == 'mean_std':
            return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def unnorm(self, x):
        if self.mode == 'minmax':
            return (x + 1) * 0.5
        elif self.mode == 'mean_std':
            return x * self.std.to(x.device) + self.mean.to(x.device)


def stopgrad(x):
    return x.detach()


def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """
    Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
    Args:
        error: Tensor of shape (B, C, W, H)
        gamma: Power used in original ||Δ||^{2γ} loss
        c: Small constant for stability
    Returns:
        Scalar loss
    """
    delta_sq = torch.mean(error ** 2, dim=(1, 2, 3), keepdim=False)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq  # ||Δ||^2
    return (stopgrad(w) * loss).mean()


class MeanFlow(nn.Module):
    def __init__(
        self,
        base_model,
        channels=1,
        image_size=32,
        normalizer=['minmax', None, None],
        # mean flow settings
        flow_ratio=0.50,
        # time distribution, mu, sigma
        time_dist=['lognorm', -0.4, 1.0],
        cfg_ratio=0.10,
        cfg_scale=2.0,
        # experimental
        cfg_uncond='u',
        jvp_api='autograd',
    ):
        super().__init__()
        self.model=base_model#定义基座模型
        self.channels = channels
        self.image_size = image_size
        self.use_cond = True

        self.normer = Normalizer.from_list(normalizer)

        self.flow_ratio = flow_ratio
        self.time_dist = time_dist
        self.cfg_ratio = cfg_ratio
        #self.w = cfg_scale
        self.w=None

        self.cfg_uncond = cfg_uncond
        self.jvp_api = jvp_api

        assert jvp_api in ['funtorch', 'autograd'], "jvp_api must be 'funtorch' or 'autograd'"
        if jvp_api == 'funtorch':
            self.jvp_fn = torch.func.jvp
            self.create_graph = False
        elif jvp_api == 'autograd':
            self.jvp_fn = torch.autograd.functional.jvp
            self.create_graph = True
        self.cond_coef = 0.1 # Condition coefficient, can be adjusted
        self.noise_coef=0.1
    # fix: r should be always not larger than t
    def sample_t_r(self, batch_size, device):
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)

        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))  # Apply sigmoid

        # Assign t = max, r = min, for each pair
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        num_selected = int(self.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r

    def loss(self, x, c):
        """
        Args:
            model: 扩散模型
            x: 真实图像 (batch_size, channels, height, width)
            c: 退化图像 (batch_size, channels, height, width)
        """
        if c is None:
            raise ValueError("退化图像 c 不能为 None")
            
        batch_size = x.shape[0]
        device = x.device

        t, r = self.sample_t_r(batch_size, device)

        t_ = rearrange(t, "b -> b 1 1 1")
        r_ = rearrange(r, "b -> b 1 1 1")

        e = torch.randn_like(x)
        x = self.normer.norm(x)  # 归一化真实图像
        c = self.normer.norm(c)  # 归一化退化图像
        #x_res=c-x

        z = (1 - t_) * x +(t_-self.cond_coef*t_)*c+self.noise_coef*t_*e#z是流路径
        v = (1-self.cond_coef)*c-x+self.noise_coef*e#瞬时速度

        if self.w is not None:
            # 对于图像修复，无条件输入可以是零图像或噪声
            uncond = torch.zeros_like(c)  # 或者 torch.randn_like(c) * 0.1
            with torch.no_grad():
                u_t = self.model(z, t, t, uncond)#网络预测平均速度
            v_hat = self.w * v + (1 - self.w) * u_t#带条件的瞬时速度
        else:
            v_hat = v

        # CFG：随机将一部分条件替换为无条件
        # cfg_mask = torch.rand(batch_size, device=device) < self.cfg_ratio
        # cfg_mask = rearrange(cfg_mask, "b -> b 1 1 1")
        # c_input = torch.where(cfg_mask, uncond, c)
        #c_input=c

        if self.cfg_uncond == 'v':
            cfg_mask_v = rearrange(r, "b -> b 1 1 1").bool()
            v_hat = torch.where(cfg_mask_v, v, v_hat)

        # forward pass 
        model_partial = partial(self.model, y=None)
        jvp_args = (
            lambda z, t, r: model_partial(z, t, r),
            (z, t, r),
            (v_hat, torch.ones_like(t), torch.zeros_like(r)),
        )#求雅可比矩阵

        if self.create_graph:
            u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
        else:
            u, dudt = self.jvp_fn(*jvp_args)

        u_tgt = v_hat - (t_ - r_) * dudt

        error = u - stopgrad(u_tgt)
        loss = adaptive_l2_loss(error)

        mse_val = (stopgrad(error) ** 2).mean()
        return loss, mse_val

    @torch.no_grad()
    def restore_images(self,degraded_images, sample_steps=1, device='cuda'):
        """
        图像修复推理函数
        Args:
            model: 训练好的模型
            degraded_images: 退化图像 (batch_size, channels, height, width)
            sample_steps: 采样步数
            device: 设备
        Returns:
            restored_images: 修复后的图像
        """
        self.model.eval()
        
        batch_size= degraded_images.shape[0]
        degraded_images = self.normer.norm(degraded_images.to(device))
        H,W=degraded_images.shape[-2:]
        # 从噪声开始
        z = torch.randn(batch_size, self.channels,
                        H, W, device=device)
        z=self.noise_coef*z+(1-self.cond_coef)*degraded_images
        t = torch.ones((batch_size,), device=device)
        r = torch.zeros((batch_size,), device=device)

        # 单步去噪（可以扩展为多步）
        u = self.model(z, t, r, y=None)
        z = z - u

        # 反归一化
        restored = self.normer.unnorm(z.clip(-1, 1))
        
        return restored
    

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def cycle(dl):
    while True:
        for data in dl:
            yield data

def exists(x):
    return x is not None
class Trainer(object):
    def __init__(
        self,
        meanflow_model,
        dataset,
        opts,
        *,
        train_batch_size=16,
        augment_flip=True,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=1000,
        results_folder='./results/sample',
        amp=False,
        fp16=False,
        split_batches=True,
        convert_image_to=None,
        condition=False,
        sub_dir=False,
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no'
        )
        self.sub_dir = sub_dir
        self.accelerator.native_amp = amp
        self.model = meanflow_model

        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size

        self.train_num_steps = train_num_steps
        self.image_size = meanflow_model.image_size
        self.condition = condition

        if self.condition:
            if opts.phase == "train":
                "论文中提到在一个批次中为各个任务设置的权重为:0.4 - 0.1 - 0.2 - 0.2 - 0.1, 即 batch_size 设置为 (4:)1 : 2: 2: 1"
                "这里可以考虑重新设置权重"
                "各任务的数据集大小差异显著：light_only: 485 张，rain: 13711 张，snow: 18069 张，blur: 2103 张 -> 4206 张（1:1的数据增强，使用helper.py）"
                "调整为：1: 4: 4: 2"
                #self.dl_fog = cycle(self.accelerator.prepare(DataLoader(dataset[0], batch_size=4, shuffle=True, pin_memory=True, num_workers=4)))
                self.dl_light = cycle(self.accelerator.prepare(DataLoader(dataset[0], batch_size=1, shuffle=True, pin_memory=True, num_workers=2)))
                self.dl_rain = cycle(self.accelerator.prepare(DataLoader(dataset[1], batch_size=2, shuffle=True, pin_memory=True, num_workers=2)))
                self.dl_snow = cycle(self.accelerator.prepare(DataLoader(dataset[2], batch_size=2, shuffle=True, pin_memory=True, num_workers=2)))
                self.dl_blur = cycle(self.accelerator.prepare(DataLoader(dataset[3], batch_size=1, shuffle=True, pin_memory=True, num_workers=2)))
            else:
                self.sample_dataset = dataset
                

        # optimizer
        self.opt0 = Adam(meanflow_model.parameters(), lr=train_lr, betas=adam_betas)

        if self.accelerator.is_main_process:
            self.ema = EMA(meanflow_model, beta=ema_decay,
                           update_every=ema_update_every)

            self.set_results_folder(results_folder)

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt0 = self.accelerator.prepare(self.model, self.opt0)
        
        device = self.accelerator.device
        self.device = device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return
        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt0': self.opt0.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        path = Path(self.results_folder / f'model-{milestone}.pt')
        if path.exists():
            data = torch.load(str(path), map_location=self.device)
            self.model = self.accelerator.unwrap_model(self.model)
  
            self.model.load_state_dict(data['model'])
            self.step = data['step']
            
            self.opt0.load_state_dict(data['opt0'])
            self.opt0.param_groups[0]['capturable'] = True
            self.ema.load_state_dict(data['ema'])

            if exists(self.accelerator.scaler) and exists(data['scaler']):
                self.accelerator.scaler.load_state_dict(data['scaler'])

            print("load model - "+str(path))

        # self.ema.to(self.device)

    def train(self):
        accelerator = self.accelerator

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0
                if self.condition:
                    #batch1 = next(self.dl_fog)
                    batch1 = next(self.dl_light)
                    batch2 = next(self.dl_rain)
                    batch3 = next(self.dl_snow)
                    batch4 = next(self.dl_blur)
                   
                    data = {}
                    for k, v in batch1.items():
                        if 'path' in k:
                            data[k] = batch1[k]+batch2[k]+batch3[k]+batch4[k] 
                        else:
                            data[k] = torch.cat([batch1[k],batch2[k],batch3[k],batch4[k]], dim=0)
                    gt = data["gt"].to(self.device)
                    cond_input = data["adap"].to(self.device)
                    task = data["A_paths"]
                    data = [gt, cond_input, task]
                else:
                    data = next(self.dl)
                    data = data[0] if isinstance(data, list) else data
                    data = data.to(self.device)

                with self.accelerator.autocast():
                    loss,mse_val = self.model.loss(data[0],data[1])
                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                accelerator.wait_for_everyone()

                self.opt0.step()
                self.opt0.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                total_loss+=loss.item()
                if accelerator.is_main_process:
                    self.ema.to(self.device)
                    self.ema.update()

                    if self.step != 0 and self.step % (self.save_and_sample_every*10) == 0:
                        milestone = self.step // self.save_and_sample_every
                        self.save(milestone)
                        
                        print("we saved model - "+str(milestone)+".pt")

                    # 每隔若干步骤增加保存频率
                    if self.step != 0 and self.step % (self.save_and_sample_every) == 5:
                        print("step " + str(self.step) + ", we saved model-999999.pt",flush=True)
                        print(f"step {self.step}, loss {total_loss:.4f}",flush=True)
                        self.save(999999) # 用 model-999999.pt 记录
                        
                pbar.set_description(f'loss_unet0: {total_loss:.4f}')
                pbar.update(1)

        accelerator.print('training complete')
    
    def test(self, sample=False, last=True, FID=False):
        #self.ema.ema_model.init()
        self.ema.to(self.device)
        print("test start")
        if self.condition:
            self.ema.ema_model.eval()
            loader = DataLoader(
                dataset=self.sample_dataset,
                batch_size=1)
            i = 0
            cnt = 0
            opt_metric = {
                'psnr': {
                    'type': 'calculate_psnr',
                    'crop_border': 0,
                    'test_y_channel': True
                    },
                'ssim': {
                    'type': 'calculate_ssim',
                    'crop_border': 0,
                    'test_y_channel': True
                    }
                }
            self.metric_results = {
                metric: 0
                for metric in opt_metric.keys()
            }
            tran = transforms.ToTensor()
            for items in loader:
                if self.condition:
                    file_ = items["A_paths"][0]
                    file_name = file_.split('/')[-4]
                else:
                    file_name = f'{i}.png'

                i += 1
                
                start_time = time.time()
                with torch.no_grad():
                    
                    data = items
                    
                    x_input_sample = data["adap"].to(self.device)
                    gt = data["gt"].to(self.device)
                    all_images_list = list(self.ema.ema_model.restore_images(x_input_sample))
                print(f"用时:{time.time()-start_time}s")
                all_images_list = [all_images_list[-1]]
                all_images = torch.cat(all_images_list, dim=0)
 
                if last:
                    nrow = 1
                else:
                    nrow = all_images.shape[0]
                save_path = str(self.results_folder / file_name)
                os.makedirs(save_path, exist_ok=True)
                full_path = os.path.join(save_path, file_.split('/')[-1]).replace('_fake_B','')
                utils.save_image(all_images, full_path, nrow=nrow)
                print("test-save "+full_path)
                
                #calculate the metric
            
                sr_img = tensor2img(all_images, rgb2bgr=True)
                gt_img = tensor2img(gt, rgb2bgr=True)
                opt_metric_ = {
                    'psnr': {
                        'type': 'calculate_psnr',
                        'crop_border': 0,
                        'test_y_channel': True
                        },
                    'ssim': {
                        'type': 'calculate_ssim',
                        'crop_border': 0,
                        'test_y_channel': True
                        }
                    }
                for name, opt_ in opt_metric_.items():
                    metric_type = opt_.pop('type')
                    self.metric_results[name] += getattr(metric_module, metric_type)(sr_img, gt_img, **opt_)
               
                cnt += 1

            current_metric = {}
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric[metric] = self.metric_results[metric]
            print(current_metric['psnr'])
            print(current_metric['ssim'])
        
        print("test end")
    def test_skipexist(self, sample=False, last=True, FID=False):
        self.ema.ema_model.init()
        self.ema.to(self.device)
        print("test start")
        if self.condition:
            self.ema.ema_model.eval()
            loader = DataLoader(
                dataset=self.sample_dataset,
                batch_size=1)
            i = 0
            cnt = 0
            opt_metric = {
                'psnr': {
                    'type': 'calculate_psnr',
                    'crop_border': 0,
                    'test_y_channel': True
                    },
                'ssim': {
                    'type': 'calculate_ssim',
                    'crop_border': 0,
                    'test_y_channel': True
                    }
                }
            self.metric_results = {
                metric: 0
                for metric in opt_metric.keys()
            }
            tran = transforms.ToTensor()
            for items in loader:
                if self.condition:
                    file_ = items["A_paths"][0]
                    file_name = file_.split('/')[-4]
                else:
                    file_name = f'{i}.png'

                i += 1
                
                start_time = time.time()
                save_path = str(self.results_folder / file_name)
                os.makedirs(save_path, exist_ok=True)
                full_path = os.path.join(save_path, file_.split('/')[-1]).replace('_fake_B','')
                with torch.no_grad():
                    batches = self.num_samples
                    
                    data = items
                    x_input_sample = data["adap"].to(self.device)
                    gt = data["gt"].to(self.device)
                    if not os.path.exists(full_path):
                        all_images_list = list(self.ema.ema_model.sample(
                                x_input_sample, batch_size=batches, last=last, task=file_))
                    else:
                        cur_image= Image.open(full_path).convert('RGB')
                        cur_image=self.sample_dataset.transform_img(cur_image)
                        all_images_list = [cur_image]
                print(f"用时:{time.time()-start_time}s")
                all_images_list = [all_images_list[-1]]
                all_images = torch.cat(all_images_list, dim=0)
 
                if last:
                    nrow = int(math.sqrt(self.num_samples))
                else:
                    nrow = all_images.shape[0]
                if not os.path.exists(full_path):
                    utils.save_image(all_images, full_path, nrow=nrow)
                
                print("test-save "+full_path)
                
                #calculate the metric
            
                sr_img = tensor2img(all_images, rgb2bgr=True)
                gt_img = tensor2img(gt, rgb2bgr=True)
                opt_metric_ = {
                    'psnr': {
                        'type': 'calculate_psnr',
                        'crop_border': 0,
                        'test_y_channel': True
                        },
                    'ssim': {
                        'type': 'calculate_ssim',
                        'crop_border': 0,
                        'test_y_channel': True
                        }
                    }
                for name, opt_ in opt_metric_.items():
                    metric_type = opt_.pop('type')
                    self.metric_results[name] += getattr(metric_module, metric_type)(sr_img, gt_img, **opt_)
               
                cnt += 1

            current_metric = {}
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric[metric] = self.metric_results[metric]
            print(current_metric['psnr'])
            print(current_metric['ssim'])
        
        print("test end")
    def set_results_folder(self, path):
        self.results_folder = Path(path)
        if not self.results_folder.exists():
            os.makedirs(self.results_folder)
