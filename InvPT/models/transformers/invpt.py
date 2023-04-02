# By Hanrong Ye
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange as o_rearrange
from einops.layers.torch import Rearrange
from utils.utils import to_2tuple
from timm.models.layers import DropPath, trunc_normal_
import pdb

BATCHNORM = nn.SyncBatchNorm # nn.BatchNorm2d

def rearrange(*args, **kwargs):
    return o_rearrange(*args, **kwargs).contiguous()

class UpEmbed(nn.Module):

    def __init__(self,
                 patch_size=7,
                 in_chans=3,
                 embed_dim=64,
                 stride=4,
                 padding=2,
                 ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.proj = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, padding=padding, stride=stride, bias=False, dilation=padding),
                    BATCHNORM(embed_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim, embed_dim, kernel_size=patch_size, padding=padding, stride=stride, bias=False, dilation=padding),
                    BATCHNORM(embed_dim),
                    nn.ReLU(inplace=True)
                    )

    def forward(self, x):
        x = self.proj(x)
        return x

class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self,
                 fea_no,
                 dim_in,
                 dim_out,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 q_method='dw_bn',
                 kv_method='dw_bn',
                 kernel_size_q=3,
                 kernel_size_kv=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 **kwargs
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads
        self.scale = dim_out ** -0.5
        self.fea_no = fea_no

        self.conv_proj_q = self._build_projection(
            dim_in, kernel_size_q, padding_q,
            stride_q, q_method 
        )
        self.conv_proj_k = self._build_projection(
            dim_in, kernel_size_kv, padding_kv,
            stride_kv, kv_method
        )
        self.conv_proj_v = self._build_projection(
            dim_in, kernel_size_kv, padding_kv,
            stride_kv, kv_method
        )

        self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

        self.fuse_attn = nn.Conv2d(num_heads*2, num_heads, 1)

    def _build_projection(self,
                          dim_in,
                          kernel_size,
                          padding,
                          stride,
                          method):
        if method == 'dw_bn':
            proj = [nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    groups=dim_in
                )),
                ('bn', BATCHNORM(dim_in)),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ])) for _ in range(self.fea_no)]
            proj = nn.ModuleList(proj)
        elif method == 'avg':
            proj = [nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool2d(
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    ceil_mode=True
                )),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ])) for _ in range(self.fea_no)]
            proj = nn.ModuleList(proj)

        elif method == 'linear':
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj

    def split_x(self, x, h, w):
        res = h*w  
        x_list = []
        for i in range(self.fea_no):
            _x = rearrange(x[:, res*i:res*(i+1), :], 'b (h w) c -> b c h w', h=h, w=w)
            x_list.append(_x)
        return x_list

    def forward_conv(self, x, h ,w):

        x_list = self.split_x(x, h, w)

        if self.conv_proj_q is not None:
            q_list = [self.conv_proj_q[i](x_list[i]) for i in range(self.fea_no)]
            q = torch.cat(q_list, dim=1)
        else:
            q_list = [rearrange(x, 'b c h w -> b (h w) c') for x in x_list]
            q = torch.cat(q_list, dim=1)

        if self.conv_proj_k is not None:
            k_list = [self.conv_proj_k[i](x_list[i]) for i in range(self.fea_no)]
            k = torch.cat(k_list, dim=1)
        else:
            k_list = [rearrange(x, 'b c h w -> b (h w) c') for x in x_list]
            k = torch.cat(k_list, dim=1)

        if self.conv_proj_v is not None:
            v_list = [self.conv_proj_v[i](x_list[i]) for i in range(self.fea_no)]
            v = torch.cat(v_list, dim=1)
        else:
            v_list = [rearrange(x, 'b c h w -> b (h w) c') for x in x_list]
            v = torch.cat(v_list, dim=1)

        return q, k, v

    def forward(self, x, h, w, messages):
        if (
            self.conv_proj_q is not None
            or self.conv_proj_k is not None
            or self.conv_proj_v is not None
        ):
            q, k, v = self.forward_conv(x, h, w)

        q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)

        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale

        # attention message passing with upsampling
        if messages['attn'] != None:
            res_attn_score = messages['attn']

            sh = h // 4 # source h
            sw = w // 4 # source w
            bs, heads, _, _ = res_attn_score.shape

            # separate task
            res_attn_score_list = []
            res = sh*sw
            for i in range(self.fea_no):
                _x = res_attn_score[:, :, res*i:res*(i+1), :]
                _x = rearrange(_x, 'b h (m n) a -> (b h) a m n', m=sh, n=sw)
                _x = F.interpolate(_x, scale_factor=2, mode='bilinear', align_corners=False)
                _x = rearrange(_x, '(b h) a m n -> b h (m n) a', b=bs, h=heads)
                res_attn_score_list.append(_x)
            res_attn_score = torch.cat(res_attn_score_list, dim=2)

            # fuse both the attention map from last stage and current stage
            multiScaleAttention = torch.cat([attn_score, res_attn_score], dim=1)
            # attn_score += res_attn_score
            attn_score = self.fuse_attn(multiScaleAttention)
        messages['attn'] = attn_score 

        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)

        x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        x = rearrange(x, 'b h t d -> b t (h d)')

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class InvPTBlock(nn.Module):

    def __init__(self,
                 task_no,
                 dim_in,
                 dim_out,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()

        self.stride_q = kwargs['stride_q']
        self.embed_dim = dim_in // task_no
        self.task_no = task_no

        dim_mlp_hidden = int(dim_out * mlp_ratio)
        self.drop_path = DropPath(drop_path) \
            if drop_path > 0. else nn.Identity()

        self.norm1 = norm_layer(self.embed_dim)
        self.norm2 = norm_layer(self.embed_dim)
        dim_mlp_hidden = int(self.embed_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=self.embed_dim,
            hidden_features=dim_mlp_hidden,
            act_layer=act_layer,
            drop=drop
        )
        self.attn = SelfAttention(task_no,
            self.embed_dim, self.embed_dim, num_heads, qkv_bias, attn_drop, drop,
            **kwargs
        )

    def split_x(self, x, h, w):
        res = h*w  
        x_list = []
        for i in range(self.task_no):
            _x = x[:, res*i:res*(i+1), :]
            x_list.append(_x)
        return x_list

    def forward(self, x_list, messages):

        h, w = x_list[0].shape[2:]
        x_list = [rearrange(_x, 'b c h w -> b (h w) c') for _x in x_list]
        x = torch.cat(x_list, dim=1) # cat on space dim
        res = x
        x = self.norm1(x)
        attn = self.attn(x, h, w, messages)

        # interpolate output of attention to previous resolution
        sh, sw = h//self.stride_q, w//self.stride_q
        attn_list = self.split_x(attn, sh, sw)
        attn_list = [rearrange(_it, 'b (h w) c -> b c h w', h=sh, w=sw) for _it in attn_list]
        attn_list = [F.interpolate(_it, size=(h, w), mode='bilinear', align_corners=False) for _it in attn_list]
        attn_list = [rearrange(_it, 'b c h w -> b (h w) c') for _it in attn_list]
        attn = torch.cat(attn_list, dim=1)

        x = res + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x_list = self.split_x(x, h, w)
        x_list = [rearrange(_it, 'b (h w) c -> b c h w', h=h, w=w) for _it in x_list]

        return x_list

class InvPTStage(nn.Module):
    def __init__(self,
                 p,
                 stage_idx,
                 patch_size=16,
                 patch_stride=16,
                 patch_padding=0,
                 in_chans=3,
                 embed_dim=768,
                 depth=1,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 **kwargs):
        super().__init__()
        assert depth == 1
        self.stage_idx = stage_idx
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.task_no = len(p.TASKS.NAMES)

        self.rearrage = None

        if patch_size == 0:
            self.patch_embed = None
        else:
            self.patch_embed = [UpEmbed(
                patch_size=patch_size,
                in_chans=in_chans,
                stride=patch_stride,
                padding=patch_padding,
                embed_dim=embed_dim,
            ) for _ in range(self.task_no)]

            self.patch_embed = nn.ModuleList(self.patch_embed)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule, but we only use depth=1 here.

        blocks = []
        for j in range(depth):
            blocks.append(
                InvPTBlock(
                    dim_in=embed_dim*self.task_no,
                    dim_out=embed_dim*self.task_no,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    **kwargs
                )
            )
        self.blocks = nn.ModuleList(blocks)

        if init == 'xavier':
            self.apply(self._init_weights_xavier)
        else:
            self.apply(self._init_weights_trunc_normal)

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, BATCHNORM)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, BATCHNORM)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_list, messages, back_fea):
        if self.patch_embed != None:
            for i in range(self.task_no):
                x = self.patch_embed[i](x_list[i])

                # backbone skip connection
                if self.stage_idx == 0:
                    x = x + back_fea[2]
                if self.stage_idx == 1:
                    x = x + back_fea[1]
                elif self.stage_idx == 2:
                    x = x + back_fea[0]
                x_list[i] = x

        for i, blk in enumerate(self.blocks):
            x_list = blk(x_list, messages)

        return x_list

class InvPT(nn.Module):
    def __init__(self,
                 p,
                 in_chans=3,
                 num_classes=1000,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 spec=None):
        super().__init__()
        self.num_classes = num_classes
        self.p = p

        self.all_tasks = p.TASKS.NAMES
        task_no = len(self.all_tasks)
        self.task_no = task_no

        self.num_stages = spec['NUM_STAGES']

        embed_dim = in_chans
        self.embed_dim = embed_dim

        mt_in_chans = embed_dim 
        self.norm_mts = nn.ModuleList()
        self.mt_embed_dims = []
        target_channel = in_chans 
        self.redu_chan = nn.ModuleList()
        self.invpt_stages = nn.ModuleList()
        for i in range(self.num_stages):
            cur_mt_embed_dim = spec['DIM_EMBED'][i] 
            kwargs = {
                'task_no': task_no,
                'patch_size': spec['PATCH_SIZE'][i],
                'patch_stride': spec['PATCH_STRIDE'][i],
                'patch_padding': spec['PATCH_PADDING'][i],
                'embed_dim': cur_mt_embed_dim, 
                'depth': 1,
                'num_heads': spec['NUM_HEADS'][i],
                'mlp_ratio': spec['MLP_RATIO'][i],
                'qkv_bias': spec['QKV_BIAS'][i],
                'drop_rate': 0,
                'attn_drop_rate': 0,
                'drop_path_rate': spec['DROP_PATH_RATE'][i],
                'q_method': spec['Q_PROJ_METHOD'][i],
                'kv_method': spec['KV_PROJ_METHOD'][i],
                'kernel_size_q': spec['KERNEL_Q'][i],
                'kernel_size_kv': spec['KERNEL_KV'][i],
                'padding_q': spec['PADDING_Q'][i],
                'padding_kv': spec['PADDING_KV'][i],
                'stride_kv': spec['STRIDE_KV'][i],
                'stride_q': spec['STRIDE_Q'][i],
            }
            stage = InvPTStage(
                p=p,
                stage_idx=i,
                in_chans=mt_in_chans,
                init=init,
                act_layer=act_layer,
                norm_layer=norm_layer,
                **kwargs
            )
            self.invpt_stages.append(stage)
            mt_in_chans = cur_mt_embed_dim 
            self.norm_mts.append(norm_layer(mt_in_chans*task_no))
            self.mt_embed_dims.append(mt_in_chans)
            _redu_chan = nn.ModuleList([nn.Conv2d(mt_in_chans, target_channel,1) for _ in range(task_no)])
            self.redu_chan.append(_redu_chan)
        self.mt_embed_dim = target_channel 
        mt_in_chans = task_no * mt_in_chans
        self.norm_mt = norm_layer(mt_in_chans)

        # Final convs
        self.mt_proj = nn.ModuleDict()
        for task in self.all_tasks:
            self.mt_proj[task] = nn.Sequential(nn.Conv2d(self.mt_embed_dim, self.mt_embed_dim, 3, padding=1), BATCHNORM(self.mt_embed_dim), nn.ReLU(True))
            trunc_normal_(self.mt_proj[task][0].weight, std=0.02)

        # combining task features and preliminary predictions
        self.mix_proj = nn.ModuleDict()
        for t in self.all_tasks:
            _mix_channel = spec['ori_embed_dim'] + p.TASKS.NUM_OUTPUT[t]
            self.mix_proj[t] = nn.Sequential(nn.Conv2d(_mix_channel, embed_dim, 1))

    def forward(self, x_dict, inter_pred, back_fea):
        '''
        Input: 
        x_dict: dict of feature map lists {task: [torch.tensor([B, H*W, embed_dim]), xxx]}
        '''
        x_list = []

        for inp_t in self.all_tasks: 
            _x = x_dict[inp_t]
            _x = torch.cat([_x, inter_pred[inp_t]], dim=1)
            _x = self.mix_proj[inp_t](_x)
            x_list.append(_x)

        messages = {'attn': None}

        h, w = self.p.mtt_resolution
        th = h * 2**(self.num_stages-1) * 2
        tw = w * 2**(self.num_stages-1) * 2
        multi_scale_task_feature = {_t: 0 for _t in self.all_tasks}

        for i in range(self.num_stages):
            x_list = self.invpt_stages[i](x_list, messages, back_fea)
            _x_list = [rearrange(_x, 'b c h w -> b (h w) c') for _x in x_list]
            x = torch.cat(_x_list, dim=2)
            x = self.norm_mts[i](x)

            nh = h * 2**(i)
            nw = w * 2**(i)
            x = rearrange(x, 'b (h w) c -> b c h w', h=nh, w=nw)

            for ii, task in enumerate(self.all_tasks):
                mt_embed_dim = self.mt_embed_dims[i]
                task_x = x[:, mt_embed_dim*ii: mt_embed_dim*(ii+1), :, :]
                if i > 0:
                    task_x = self.redu_chan[i][ii](task_x)
                task_x = F.interpolate(task_x, size=(th, tw), mode='bilinear', align_corners=False)
                # add feature from all the scales
                multi_scale_task_feature[task] += task_x

        x_dict = {}
        for i, task in enumerate(self.all_tasks):
            x_dict[task] = self.mt_proj[task](multi_scale_task_feature[task])

        return  x_dict