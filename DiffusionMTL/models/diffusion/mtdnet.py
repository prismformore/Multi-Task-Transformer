import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from functools import partial

from .util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config, extract_into_tensor, make_beta_schedule, noise_like, detach
import os
from utils.common_config import get_head
INTERPOLATE_MODE = 'bilinear'

def tolist(a):
    try:
        return [tolist(i) for i in a]
    except TypeError:
        return a

from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    from DIT
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, kv):
        B, N, C = x.shape
        _, T, _ = kv.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # (B, heads, N, C//heads)
        kv = self.kv(kv).reshape(B, T, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # (2, B, heads, T, C//heads)
        k, v = kv[0], kv[1]# (B, heads, T, C//heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale # (B, heads, N, T), N is 1
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.mlp_kv = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm_kv = norm_layer(dim)
        self.norm2_kv = norm_layer(dim)

    def forward(self, x, kv):
        kv = kv + self.drop_path(self.mlp_kv(self.norm2_kv(kv)))

        x = x + self.drop_path(self.attn(self.norm1(x), self.norm_kv(kv)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, kv

class Denosier(nn.Module):
    def __init__(self, p, embed_dim):
        super().__init__()
        out_dim = p.backbone_channels[-1]
        self.tasks = p.TASKS.NAMES

        self.depth = 4
        self.crossblocks = nn.ModuleList([CrossBlock(embed_dim, num_heads=4) for _ in range(self.depth)])

        self.pre_layer = nn.ModuleDict({t: nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1) for t in self.tasks})

        self.task_heads = nn.ModuleDict({task: nn.Sequential(nn.Conv2d(embed_dim, embed_dim, kernel_size=1), 
                                        nn.Conv2d(embed_dim, 512, kernel_size=3, padding=1), nn.ReLU(),
                                        nn.Conv2d(512, out_dim, kernel_size=3, padding=1), nn.ReLU(),
                                        nn.Conv2d(out_dim, out_dim, 3, padding=1), nn.ReLU(),
                                        )
                                        for task in p.TASKS.NAMES})
        

        self.final_task_heads = nn.ModuleDict(
            {task: get_head(p, p.backbone_channels, task) for task in p.TASKS.NAMES}
        )

        self.initialize_weights()

        self.t_embedder = TimestepEmbedder(embed_dim)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)
        self.apply(_basic_init)


    def forward(self, feat, noisy_masks, info, upsam=False, t=None):

        out = {}
        tasks = list(noisy_masks.keys()) # should be only one task for now 
        t = self.t_embedder(t)[:,:,None,None]

        resize_scale = 0.25
        cur_feat = F.interpolate(feat, scale_factor=resize_scale, mode='bilinear', align_corners=False)
        B, C, H, W = cur_feat.shape
        kv = cur_feat.reshape(B, C, -1).transpose(1,2) # (B, N, C)

        for task_idx, task in enumerate(tasks):
            x = noisy_masks[task]
            x = self.pre_layer[task](x)
            x = x + t
            x0 = x
            cur_kv = kv

            x = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)
            B, C, H, W = x.shape
            x = x.reshape(B,C,-1).transpose(1,2)

            for _d in range(self.depth):
                cur_kv, x = self.crossblocks[_d](cur_kv, x)
            x = cur_kv

            x = x.reshape(B, H, W, C).permute(0,3,1,2)
            x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
            x = x + x0

            task_x = x
            task_mask = self.task_heads[task](task_x)
            if upsam:
                img_size = info['img_size']
                task_mask = self.final_task_heads[task](task_mask)
                task_mask = F.interpolate(task_mask, img_size, mode=INTERPOLATE_MODE, align_corners=False)
            out[task] = task_mask
        
        return out

class FeatureCombinator(nn.Module):
    def __init__(self, p, embed_dim, inp_chan) -> None:
        super().__init__()
        tasks = p.TASKS.NAMES
        self.out_layer = nn.Conv2d(inp_chan, embed_dim, kernel_size=3, padding=1)
        self.initialize_weights()

    def forward(self, x):
        x = self.out_layer(x)
        return x

    def initialize_weights(self):
        # Initialize transformer or CNN layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

class MTDNet(nn.Module):
    def __init__(self, p, embed_dim):
        super().__init__()
        ### hyper-parameters
        # self.parameterization = 'eps'
        self.parameterization = 'x0'
        beta_schedule="linear"
        self.num_timesteps = p.num_timesteps
        timesteps = p.num_timesteps
        linear_start=1e-3
        linear_end=1e-2
        self.v_posterior = v_posterior =0 
        self.loss_type="l2"
        self.signal_scaling_rate = 1
        ###

        self.p = p
        self.tasks = p.ALL_TASKS.NAMES
        new_embed_dim = embed_dim
        self.denoiser = Denosier(p, new_embed_dim)


        # q sampling
        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        # diffusion loss
        learn_logvar = False
        logvar_init = 0
        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(p.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)
        self.l_simple_weight = 1.

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        if len(lvlb_weights) > 1:
            lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

        self.pred_combinator = FeatureCombinator(p, embed_dim, sum(self.p.TASKS.NUM_OUTPUT.values()))


    def q_sample(self, x_start, t, noise=None):
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, feat, info, upsam, noisy_masks, t, clip_denoised: bool):
        tasks = list(noisy_masks.keys())
        model_out = self.gen_pred(feat, noisy_masks, info, upsam, t)

        model_mean_dict, posterior_variance_dict, posterior_log_variance_dict = {}, {}, {}
        for task in tasks:
            task_model_out = model_out[task]
            x = noisy_masks[task]
            if self.parameterization == "eps":
                x_recon = self.predict_start_from_noise(x, t=t, noise=task_model_out)
            elif self.parameterization == "x0":
                x_recon = task_model_out
            if clip_denoised:
                x_recon.clamp_(-1., 1.)

            if upsam: # last sampling step
                model_mean = x_recon
                posterior_variance, posterior_log_variance = 0, 0
            else:
                x_recon = F.interpolate(x_recon, x.shape[2:], mode=INTERPOLATE_MODE, align_corners=False)
                model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t) # Why do we need this?
            model_mean_dict[task] = model_mean
            posterior_variance_dict[task] = posterior_variance
            posterior_log_variance_dict[task] = posterior_log_variance
        return model_mean_dict, posterior_variance_dict, posterior_log_variance_dict, model_out


    def p_sample(self, feat, noisy_masks, t, info, upsam, clip_denoised=False, repeat_noise=False):
        tasks = list(noisy_masks.keys())
        model_mean_dict, _, model_log_variance_dict, model_out_dict = self.p_mean_variance(feat, info, upsam, noisy_masks, t=t, clip_denoised=clip_denoised)

        out = {}
        for task in tasks:
            x = noisy_masks[task]
            b, *_, device = *x.shape, x.device
            model_mean = model_mean_dict[task]
            model_log_variance = model_log_variance_dict[task]
            noise = noise_like(x.shape, device, repeat_noise)
            # no noise when t == 0
            nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
            if not upsam:
                out[task] = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise #, model_out
            else:
                out[task] = model_mean

        return out

    def gen_pred(self, feat, noisy_masks, info, upsam=False, t=None):
        model_out = self.denoiser(feat, noisy_masks, info, upsam, t)
        return model_out

    def p_sample_loop(self, feat, noisy_masks, latent_shape, info):
        b = latent_shape[0]
        num_timesteps = self.num_timesteps 
        
        for t in reversed(range(0, num_timesteps)):
            for task in self.tasks:
                noisy_masks[task] = F.interpolate(noisy_masks[task], latent_shape[2:], mode=INTERPOLATE_MODE, align_corners=False)
            noisy_masks = self.p_sample(feat, noisy_masks, torch.full((b,), t, device=feat.device, dtype=torch.long), 
                                                info, upsam=True if t==0 else False)
        return noisy_masks

    def forward(self, feature, info, batch):
        inter_feas, inter_preds = feature

        inter_preds = {k: v.detach() for k, v in inter_preds.items()}

        combined_pred = torch.cat([_ for _ in inter_preds.values()], dim=1)
        combined_pred = self.pred_combinator(combined_pred)

        if self.training:
            info['noisy_masks'] = [] # [{task: mask in available task} x bs]
            info['t'] = []
            info['model_out_dict'] = []
            info['model_mean_dict'] = []
            info['sampled_mask_dict'] = []
            info['available_tasks'] = []
            all_noisy_masks = []
            bs = list(inter_feas.values())[0].shape[0]
            for b_idx in range(bs):
                info['model_out_dict'].append([])
                info['model_mean_dict'].append([])
                info['sampled_mask_dict'].append([])
                info['noisy_masks'].append({})
                info['t'].append({})
                all_noisy_masks.append({})
                x = combined_pred[b_idx:b_idx+1]
                available_tasks = [] # tasks with labels
                for _t_idx, _task in enumerate(self.tasks):
                    if batch['task_w'][b_idx][_t_idx]:
                        available_tasks.append(_task)

                info['available_tasks'].append(available_tasks)

                for task in available_tasks:
                    noisy_masks = {}
                    latent_shape = x.shape
                    x_start = inter_feas[task][b_idx:b_idx+1] * self.signal_scaling_rate #batch['mask_'+task]
                    t = torch.full((x.shape[0],), self.num_timesteps-1, device=x.device, dtype=torch.long)
                    noise = default(None, lambda: torch.randn_like(x_start))
                    _x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                    noisy_masks[task] = _x_noisy
                    for _t in reversed(range(1, self.num_timesteps)):
                        _t = torch.full((x.shape[0],), _t, device=x.device, dtype=torch.long)
                        noisy_masks = self.p_sample(x, noisy_masks, _t, info, upsam=False)
                    # detach(noisy_masks)
                    _t = 0
                    _t = torch.full((x.shape[0],), _t, device=x.device, dtype=torch.long)
                    noisy_masks = self.p_sample(x, noisy_masks, _t, info, upsam=True)
                    all_noisy_masks[b_idx][task] = noisy_masks[task]

        else:
            info['noisy_masks'] = {} # {task: mask_of_batch}
            info['model_out_dict'] = [] # [{task: mask_of_batch in all tasks} x time_steps]
            info['model_mean_dict'] = []
            info['sampled_mask_dict'] = []
            noisy_masks = {}
            x = combined_pred
            for task in self.tasks:
                latent_shape = x.shape
                x_start = inter_feas[task] * self.signal_scaling_rate #batch['mask_'+task]
                noise = default(None, lambda: torch.randn_like(x_start))
                t = torch.ones(x.shape[0], device=x.device).long() * (self.num_timesteps - 1)
                _x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                noisy_masks[task] = _x_noisy

            noisy_masks = self.p_sample_loop(x, noisy_masks, x.shape, info)
            all_noisy_masks = noisy_masks

        return all_noisy_masks, info

    def p_losses(self, model_output_list, target):
        loss_dict = {}
        prefix = '' #'train' if self.training else 'val'
        total_loss = 0
        for t in range(self.num_timesteps):
            model_output = model_output_list[t]
            loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
            logvar_t = self.logvar[t].to(target.device)
            loss = loss_simple / torch.exp(logvar_t) + logvar_t
            if self.learn_logvar:
                loss_dict.update({f'{prefix}/diff_loss_gamma': loss.mean()})
                loss_dict.update({'diff_logvar': self.logvar.data.mean()})

            loss = self.l_simple_weight * loss.mean()

            total_loss += loss
        loss_dict.update({'diff_loss': total_loss})

        return loss_dict

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss
