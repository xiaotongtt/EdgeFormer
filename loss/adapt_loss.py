from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model import common
from model import attention
import torch.nn as nn
import torch
from einops import rearrange

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    # windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size) # (B*HW, C, window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(B, -1, C, window_size,
                                                            window_size)  # (B, HW, C, window_size, window_size)

    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class ADAPT(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(ADAPT, self).__init__()
        n_feats = 64
        kernel_size = int(3)
        scale = args.scale[0]
        self.scale = scale
        self.l1 = nn.L1Loss(reduction='none')

    def shiftwindow(self, x, chunk_size, direct=0):  # 0, 1 : right
        if direct == 0:
            return x
        # print('shift')
        return torch.cat([x[chunk_size // 2:, ...], x[: chunk_size // 2, ...]], dim=0)

    def reversed_shiftwindow(self, x, chunk_size, direct=0):  # 0, 1 : right
        if direct == 0:
            return x
        # elif direct == 1:
        # print('rev shift')
        return torch.cat([x[-chunk_size // 2:, ...], x[: -chunk_size // 2, ...]], dim=0)

    def group(self, x_vis, chunk_size, direct=0):
        ##### 对x_vis进行分组, 分成B组
        N_vis, C, w_d, w_d = x_vis.shape

        if chunk_size > 0 and N_vis // chunk_size > 0:
            padding = chunk_size - N_vis % chunk_size if N_vis % chunk_size != 0 else 0
            x_vis = self.shiftwindow(x_vis, chunk_size, direct=direct)
            if padding:
                pad_x = x_vis[-padding:, ...].clone()
                x_vis = torch.cat([x_vis, pad_x], dim=0)

            n_group = (N_vis + padding) // chunk_size
            x_vis = x_vis.reshape(n_group, chunk_size, C, w_d, w_d)
        else:
            x_vis = x_vis.unsqueeze(0)

        return x_vis

    def ungroup(self, x_vis, N_vis, chunk_size, direct=0):
        _, _, C, w_d, w_d = x_vis.shape
        if chunk_size > 0 and N_vis // chunk_size > 0:
            x_vis = x_vis.reshape(-1, C, w_d, w_d)
            x_vis = x_vis[:N_vis, ...].contiguous()
            x_vis = self.reversed_shiftwindow(x_vis, chunk_size, direct=direct)
        else:
            x_vis = x_vis.reshape(-1, C, w_d, w_d)

        return x_vis

    def recover_to_origin(self, x, x_vis, mask, size):
        B, HW, C, window_size, window_size = x.shape  # (B, HW, C, window_size, window_size)
        # print(x.shape)
        H, W = size
        x = x.reshape(-1, C, window_size, window_size)
        y = torch.zeros_like(x)
        # mask = mask.reshape(-1)
        y[mask != 0.0, ...] = x_vis
        y[mask == 0.0, ...] = x[mask == 0.0, ...]
        # print(y.shape)
        y = rearrange(y, '(b h w) c p1 p2   -> b c (h p1) (w p2)', h=H, w=W)
        return y

    def forward(self, sr, hr, qk, mask, p=1):
        mask = mask.detach()
        if torch.sum(mask) == 0:
            return 0

        B, C, H, W = hr.shape
        window_size = p * self.scale
        hr_patch_f = window_partition(hr, window_size=window_size).reshape(-1, C, window_size, window_size)  # (B*HW, C, window_size, window_size)

        sr_patch_f = window_partition(sr, window_size=window_size).reshape(-1, C, window_size, window_size)

        mask = rearrange(mask, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=1, p2=1)
        mask = mask.reshape(-1)  # B*HW

        hr_patch_vis = hr_patch_f[mask != 0.0, ...]  # (N_vis, C, window_size, window_size)
        sr_patch_vis = sr_patch_f[mask != 0.0, ...]  # (N_vis, C, window_size, window_size)

        N_vis, _, _, _ = hr_patch_vis.shape
        chunk_size = min(N_vis // B, 512)

        direct = 0
        loss = 0
        for qk_att in qk:  # qk -> (B_g x chunk_size x chunk_size)
            qk_att = qk_att.detach()
            diagonal = torch.eye(qk_att.shape[1], device=qk_att.device)[None, :, :]
            qk_att.masked_fill_(diagonal, 0.0)  ### 排除对角元素

            hr_patch_vis_g = self.group(hr_patch_vis, chunk_size, direct)  # B_g x chunk_size x w_d x w_d x C
            sr_patch_vis_g = self.group(sr_patch_vis, chunk_size, direct)  # B_g x chunk_size x w_d x w_d x C
            # hr_patch_vis_g = torch.einsum('bcwed,bch->bhwed', [hr_patch_vis_g, qk_att]) # B_g x chunk_size x w_d x w_d x C

            hr_patch_vis_g = hr_patch_vis_g.unsqueeze(1)
            hr_patch_cor = hr_patch_vis_g.expand(-1, chunk_size, -1, -1, -1, -1)

            max_num = max(int(chunk_size * 0.2), 20)
            pro, max_idx = torch.sort(qk_att, descending=True, dim=2)
            idx = max_idx[:, :, :max_num]

            sr_patch_vis_g = sr_patch_vis_g.unsqueeze(2).expand(-1, -1, max_num, -1, -1, -1)

            hr_patch_vis_max = hr_patch_cor.gather(2, idx[:, :, :, None, None, None].expand(-1, -1, -1, C, window_size, window_size)) # B_g x chunk_size x num x C x w_d x w_d
            loss += torch.mean(self.l1(sr_patch_vis_g, hr_patch_vis_max) * pro[:, :, :max_num, None, None, None])

            direct = direct ^ 1

        return loss / len(qk_att)




