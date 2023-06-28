from model import common
import torch.nn as nn
import torch
import torch.nn.functional as F

class SobelConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=True, requires_grad=True):
        assert kernel_size % 2 == 1, 'SobelConv2d\'s kernel_size must be odd.'
        assert out_channels % 4 == 0, 'SobelConv2d\'s out_channels must be a multiple of 4.'
        assert out_channels % groups == 0, 'SobelConv2d\'s out_channels must be a multiple of groups.'

        super(SobelConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # In non-trainable case, it turns into normal Sobel operator with fixed weight and no bias.
        self.bias = bias if requires_grad else False

        if self.bias:
            self.bias = nn.Parameter(torch.zeros(size=(out_channels,), dtype=torch.float32), requires_grad=True)
        else:
            self.bias = None

        self.sobel_weight = nn.Parameter(torch.zeros(
            size=(out_channels, int(in_channels / groups), kernel_size, kernel_size)), requires_grad=False)

        # Initialize the Sobel kernal
        kernel_mid = kernel_size // 2
        for idx in range(out_channels):
            if idx % 4 == 0:
                self.sobel_weight[idx, :, 0, :] = -1
                self.sobel_weight[idx, :, 0, kernel_mid] = -2
                self.sobel_weight[idx, :, -1, :] = 1
                self.sobel_weight[idx, :, -1, kernel_mid] = 2
            elif idx % 4 == 1:
                self.sobel_weight[idx, :, :, 0] = -1
                self.sobel_weight[idx, :, kernel_mid, 0] = -2
                self.sobel_weight[idx, :, :, -1] = 1
                self.sobel_weight[idx, :, kernel_mid, -1] = 2
            elif idx % 4 == 2:
                self.sobel_weight[idx, :, 0, 0] = -2
                for i in range(0, kernel_mid + 1):
                    self.sobel_weight[idx, :, kernel_mid - i, i] = -1
                    self.sobel_weight[idx, :, kernel_size - 1 - i, kernel_mid + i] = 1
                self.sobel_weight[idx, :, -1, -1] = 2
            else:
                self.sobel_weight[idx, :, -1, 0] = -2
                for i in range(0, kernel_mid + 1):
                    self.sobel_weight[idx, :, kernel_mid + i, i] = -1
                    self.sobel_weight[idx, :, i, kernel_mid + i] = 1
                self.sobel_weight[idx, :, 0, -1] = 2

        # Define the trainable sobel factor
        if requires_grad:
            self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=True)
        else:
            self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=False)

    def forward(self, x):
        if torch.cuda.is_available():
            self.sobel_factor = self.sobel_factor.cuda()
            if isinstance(self.bias, nn.Parameter):
                self.bias = self.bias.cuda()

        sobel_weight = self.sobel_weight * self.sobel_factor

        if torch.cuda.is_available():
            sobel_weight = sobel_weight.cuda()

        out = F.conv2d(x, sobel_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return out
    
class SEPS(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(SEPS, self).__init__()

        n_feats = args.n_feats
        kernel_size = int(3)
        scale = args.scale[0]
        self.scale = scale

        # self.var_conv = nn.Sequential(
        #     *[conv(n_feats, n_feats, kernel_size), nn.ELU(),
        #       conv(n_feats, n_feats, kernel_size), nn.ELU(),
        #       conv(n_feats, 2, kernel_size)])
        
        self.var_conv = nn.Sequential(*[
            SobelConv2d(n_feats, n_feats), nn.ELU(),
            conv(n_feats, n_feats, kernel_size),
            conv(n_feats, 2, kernel_size)
        ])


    def flops(self):
        B, C, H, W = self.shape
        return B*C*H*W*3*3*C * 2 + B*C*H*W*2*3*3

    def forward(self, x):

        feat = x
        self.shape = x.shape
        var_f = self.var_conv(feat)

        if self.training:
            var_f = torch.nn.functional.gumbel_softmax(var_f, dim=1, tau=0.8, hard=True)
            B, C, H, W = var_f.shape
            var_f = var_f.view(B, C, -1).permute(0, 2, 1).contiguous()
            matric = torch.tensor([[0.0], [1.0]])
            var_f = torch.einsum('bhc,ci->bhi', [var_f, matric.type_as(x).to(x.device).detach()])  # B， HW， 1
            var_f = var_f.permute(0, 2, 1).reshape(B, -1, H, W)
 
        else:
            var_f = var_f.argmax(dim=1, keepdim=True).float()

        return var_f
