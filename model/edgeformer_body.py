import torch
import torch.nn as nn
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
import torch.nn.functional as F
from model import common
import numpy as np
from einops import rearrange
import random

def recover_to_origin(x, mask):
    B, _ = mask.shape
    N, S = x.shape
    mask = mask.reshape(-1)
    idx_0 = torch.nonzero(mask == 0).reshape(1, -1)
    idx_1 = torch.nonzero(mask == 1).reshape(1, -1)

    idx = torch.cat([idx_1, idx_0], dim=1).reshape(-1)

    _, unsorted_idx = idx.sort(-1)

    y = batched_index_select(x, unsorted_idx)
    return y

def batched_index_select(values, indices):
    _, s = values.shape
    return values.gather(0, indices[:, None].expand(-1, s))

def generate_mask(var, patch_size, image_size, ratio=0.6):
    H, W = image_size
    var = rearrange(var, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
    B, L, S = var.shape
    var = torch.mean(var, dim=-1)  # (B, L)
    var_mean = torch.mean(var)
  
    mask = ((var - var_mean) >= 0.0).bool()
    mask_origin = mask.unsqueeze(-1)
    mask_origin = mask_origin.expand(-1, -1, S)
    mask_origin = rearrange(mask_origin, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=H//patch_size)

    return mask, mask_origin.float()

# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''
    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

def extend_test_sinusoid_encoding_table(sinusoid_table, n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    n_origin = sinusoid_table.shape[1]
    new_sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_origin, n_position)])
    new_sinusoid_table[:, 0::2] = np.sin(new_sinusoid_table[:, 0::2])  # dim 2i
    new_sinusoid_table[:, 1::2] = np.cos(new_sinusoid_table[:, 1::2])  # dim 2i+1
    new_sinusoid_table = torch.FloatTensor(new_sinusoid_table).unsqueeze(0).to(sinusoid_table.device)

    sinusoid_table = torch.cat([sinusoid_table, new_sinusoid_table], dim=1)
    return sinusoid_table

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=192, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Multi_PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=48, patch_size=1,  in_chans=64, embed_dim=128):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        window = {1: 1,
                  3: 1,
                  5: 1,
                  7: 1}
        self.window = window
        self.conv_list = nn.ModuleList()
        self.head_splits = []
        inter_media_embed_dim = 32 # 48
        for cur_window, cur_head_split in window.items():
            dilation = 1  # Use dilation=1 at default.
            if cur_window == 5:
                n_conv = cur_window // 3 + 1
                cur_window = 3
                padding_size = (cur_window // 2)
                cur_conv = nn.Sequential(*[nn.Conv2d(
                    in_chans,
                    embed_dim // 4,
                    kernel_size=(cur_window, cur_window),
                    padding=(padding_size, padding_size),
                    dilation=(dilation, dilation),

                ), nn.Conv2d(
                    embed_dim // 4,
                    embed_dim // 4,
                    kernel_size=(cur_window, cur_window),
                    padding=(padding_size, padding_size),
                    dilation=(dilation, dilation),
                   
                )])

            elif cur_window == 7:
                n_conv = cur_window // 3 + 1
                cur_window = 3
                padding_size = (cur_window // 2)
                cur_conv = nn.Sequential(*[nn.Conv2d(
                    in_chans,
                    embed_dim // 4,
                    kernel_size=(cur_window, cur_window),
                    padding=(padding_size, padding_size),
                    dilation=(dilation, dilation),
                   
                ), nn.Conv2d(
                    embed_dim // 4,
                    embed_dim // 4,
                    kernel_size=(cur_window, cur_window),
                    padding=(padding_size, padding_size),
                    dilation=(dilation, dilation),
                  
                ), nn.Conv2d(
                    embed_dim // 4,
                    embed_dim // 4,
                    kernel_size=(cur_window, cur_window),
                    padding=(padding_size, padding_size),
                    dilation=(dilation, dilation),
                    
                )])

            else:
                padding_size = (cur_window // 2)
                cur_conv = nn.Sequential(*[nn.Conv2d(
                    in_chans,
                    embed_dim // 4,
                    kernel_size=(cur_window, cur_window),
                    padding=(padding_size, padding_size),
                    dilation=(dilation, dilation),
                  
                )])

            self.conv_list.append(cur_conv)

    def forward(self, x, **kwargs):
        conv_list = [
            conv(x) for conv in self.conv_list
        ]

        conv_img = torch.cat(conv_list, dim=1) #
        conv_img = rearrange(conv_img, "B C H W -> B (H W) C")
        return conv_img

    def flops(self, B):
        img_size = self.img_size[0]

        block_flops = dict(
            conv1=img_size * img_size* 64 * 32 * 1 * 1 * B, # 48*48*1*1*64*128
            conv2=img_size * img_size* 64 * 32 * 3 * 3 * B,
            conv3=img_size * img_size* 64 * 32 * 5 * 5 * B
        )
        return sum(block_flops.values())


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim

        head_dim = dim // num_heads # 128*4
        self.head_dim = head_dim
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads #
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias) #
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def flops(self, shape):
        dim = self.dim #128
        h = self.num_heads
        head_dim = dim // h
        B, N, C = shape

        qvk = B * N * dim * head_dim * 3 # input * output
        attention = B*h*head_dim*N*N + B*h*head_dim*N*N # q@k, k@v
        proj = B * N * dim * dim * head_dim

        return qvk + attention + proj


class DropPath(nn.Module):
   
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.o_f = out_features
        self.i_f = hidden_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def flops(self, shape):
        B, N, C = shape
        # fc1, fc2
        flops = self.o_f * self.i_f * B * N + self.o_f * self.i_f * B * N
        return flops

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
      
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.dim = dim
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

    def flops(self, shape):
        flops = 0

        flops += self.attn.flops(shape)
        flops += self.mlp.flops(shape)   # mlp
        return flops

class Encoder(nn.Module):
   
    def __init__(self, img_size=48, patch_size=1, in_chans=64, num_classes=0, embed_dim=128, depth=3, batch_size=16, chunk_size=256,
                 num_heads=4, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=0.,
                 use_learnable_pos_emb=False):
        super().__init__()
        
        self.num_classes = patch_size * patch_size * in_chans
        self.depth = depth
        self.num_features = self.embed_dim = embed_dim  
        self.chunk_size = chunk_size
       
        self.patch_embed = Multi_PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
       
        self.img_size = self.patch_embed.img_size
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
       
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:

            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)
            self.test_pos_embed = None
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        self.shape = None
        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        # trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def add_adjacent_buckets(self, x):
        x_extra_back = torch.cat([x[-1:, :self.chunk_size//2, ...], x[:-1, :self.chunk_size//2, ...]], dim=0)
        x_extra_forward = torch.cat([x[1:, self.chunk_size//2:, ...], x[:1, self.chunk_size//2:, ...]], dim=0)
        return torch.cat([x, x_extra_back, x_extra_forward], dim=1)

    def shiftwindow(self, x, chunk_size, direct=0): # 0, 1 : right
        if direct == 0:
            return x
        # print('shift')
        return torch.cat([x[chunk_size//2: , ...], x[: chunk_size//2, ...]], dim=0)

    def reversed_shiftwindow(self, x, chunk_size, direct=0): # 0, 1 : right
        if direct == 0:
            return x
        return torch.cat([x[-chunk_size//2: , ...], x[: -chunk_size//2, ...]], dim=0)

    def group(self, x_vis, chunk_size, direct=0):
        ##### 对x_vis进行分组, 分成B组
        N_vis, C = x_vis.shape

        if chunk_size > 0 and N_vis // chunk_size > 0:
            padding = chunk_size - N_vis % chunk_size if N_vis % chunk_size != 0 else 0
            x_vis = self.shiftwindow(x_vis, chunk_size, direct=direct)
            if padding:
                pad_x = x_vis[-padding:, :].clone()
                x_vis = torch.cat([x_vis, pad_x], dim=0)

            n_group = (N_vis + padding) // chunk_size
            x_vis = x_vis.reshape(n_group, chunk_size, C)
        else:
            x_vis = x_vis.unsqueeze(0)

        return x_vis

    def ungroup(self, x_vis, N_vis, chunk_size, direct=0):
        _, _, C = x_vis.shape
        if chunk_size > 0 and N_vis // chunk_size > 0:
            x_vis = x_vis.reshape(-1, C)
            x_vis = x_vis[:N_vis, :].contiguous()
            x_vis = self.reversed_shiftwindow(x_vis, chunk_size, direct=direct)

        else:
            x_vis = x_vis.reshape(-1, C)

        return x_vis

    def forward_features(self, x, mask):
        x = self.patch_embed(x)
        B, HW, C = x.shape
        self.B = B

        x = x.reshape(-1, C)
        mask = mask.reshape(-1)  # (-1)
        x = x * mask.unsqueeze(-1)

        x_vis = x[mask != 0.0, ...].reshape(-1, C)
        x_unvis = x[mask == 0.0, ...].reshape(-1, C)

        N_vis, _ = x_vis.shape
        chunk_size = min(N_vis // B, 512)
        direct = 0

        for blk in self.blocks:
            x_vis = self.group(x_vis, chunk_size, direct)
            self.shape = x_vis.shape
            x_vis = blk(x_vis)

            x_vis = self.ungroup(x_vis, N_vis, chunk_size, direct)
            direct = direct ^ 1

        x_vis = self.norm(x_vis)

        return x_vis, x_unvis, mask.reshape(B, HW)

    def forward(self, x, mask):
        x_vis, x_unvis, mask = self.forward_features(x, mask)  # B, L, 128
        return x_vis, x_unvis, mask

    def flops(self):
        flops = 0

        img_size = self.img_size[0]
        B = self.B
        patch_flops = dict(
            conv1=img_size * img_size * 64 * 32 * 1 * 1 * B,  # 48*48*1*1*64*128
            conv2=img_size * img_size * 64 * 32 * 3 * 3 * B,
            conv3=img_size * img_size * 64 * 32 * 5 * 5 * B
        )
        flops += sum(patch_flops.values())

        # attn
        dim = self.embed_dim  # 128
        h = self.num_heads
        head_dim = dim // h
        B, N, C = self.shape

        qvk = B * N * dim * head_dim * 3  # input * output
        attention = B * h * head_dim * N * N + B * h * head_dim * N * N  # q@k, k@v
        proj = B * N * dim * dim * head_dim

        attn_flops = qvk + attention +proj
        # mlp
        B, N, C = self.shape
        
        # fc1, fc2
        mlp_flops = self.embed_dim* self.mlp_ratio * self.embed_dim * B * N + self.embed_dim*self.mlp_ratio * self.embed_dim * B * N
        block_flops = attn_flops+mlp_flops

        flops += block_flops * self.depth
        return flops

class EdgeformerBody(nn.Module):
    def __init__(self, args, img_size=48, patch_size=1, n_feats=64, depth=3, chunk_size=256, conv=common.default_conv):
        super(EdgeformerBody, self).__init__()
        self.patch_size = patch_size
        encoder_embed_dim = 128
        decoder_embed_dim = n_feats
        self.dim = encoder_embed_dim
        self.encoder = Encoder(in_chans=n_feats, img_size=img_size, embed_dim=encoder_embed_dim, patch_size=patch_size, depth=depth, chunk_size=chunk_size)
        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

    def generate_mask(self, mask, patch_size=1):
        mask = rearrange(mask, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
        return mask.squeeze(-1)

    def forward(self, input):
        x, mask = input
        B, C, H, W = x.shape
        patch_size = self.patch_size

        mask = self.generate_mask(mask, patch_size)  ### 全部都输入进去

        if torch.sum(mask) == 0:
            self.set_flops(0)
            return x, self.flops

        if torch.sum(mask != 1.0) == 0:
            index = torch.randperm(mask.shape[1])[:mask.shape[1]//2]
            mask[:, index] = 0.0  ### 随机选择一半

        x_vis, x_unvis, _ = self.encoder(x, mask)  # [N_vis, C_e]
        x_full = torch.cat([x_vis, x_unvis], dim=0)
        x_full = self.encoder_to_decoder(x_full)

        y = recover_to_origin(x_full, mask)

        rec_img = rearrange(y, '(b n) c -> b n c', b=B)
        _, HW, _ = rec_img.shape
     
        y = rearrange(rec_img, 'b (h w) (p1 p2 c)  -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size,
                      h=H // patch_size, w=W // patch_size)

        return y + x


    def set_flops(self, x):
        self.flops = x

    def get_flops(self):
        return self.flops