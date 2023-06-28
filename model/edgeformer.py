import torch
import utility
import torch.optim as optim
from model import common
from model.seps import SEPS
from model.edgeformer_body import EdgeformerBody
import torch.nn as nn
import torch.nn.parallel as P


def make_model(args, parent=False):
    if args.dilation:
        from model import dilated
        return EDSR(args, dilated.dilated_conv)
    else:
        return EDSR(args)

class EDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__()

        n_resblock = args.n_resblocks # 16
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        self.scale = scale
        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.error_model_num = 2
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.n_GPUs = args.n_GPUs
        chunk_size = args.window_size
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_seps = [SEPS(args)]
        m_body_1 = [EdgeformerBody(args, img_size=48, n_feats=n_feats, patch_size=1, depth=args.refinetor_deep, chunk_size=chunk_size)]
        for i in range(n_resblock // 2):
            m_body_1.append(common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale))

        m_body_2 = [EdgeformerBody(args, img_size=48, n_feats=n_feats, patch_size=1, depth=args.refinetor_deep, chunk_size=chunk_size)]
        for i in range(n_resblock // 2):
            m_body_2.append(common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale))

        m_body_3 = [EdgeformerBody(args, img_size=48, n_feats=n_feats, patch_size=1, depth=args.refinetor_deep, chunk_size=chunk_size)]
        m_body_conv = [conv(n_feats, n_feats, kernel_size)]
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            nn.Conv2d(
                n_feats, args.n_colors, kernel_size,
                padding=(kernel_size//2)
            )
        ]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body_1 = nn.Sequential(*m_body_1)
        self.body_2 = nn.Sequential(*m_body_2)
        self.body_3 = nn.Sequential(*m_body_3)
        self.body_conv = nn.Sequential(*m_body_conv)
        self.seps = nn.Sequential(*m_seps)
        self.tail = nn.Sequential(*m_tail)
        self.max_pool = nn.MaxPool2d(scale, stride=scale)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = x

        seps_1 = self.seps(res.detach())
        res = self.body_1((res, seps_1))

        seps_2 = self.seps((x + res).detach())
        res = self.body_2((res, seps_2))

        seps_3 = self.seps((x + res).detach())
        res = self.body_3((res, seps_3))

        res = self.body_conv(res)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x, [seps_1, seps_2, seps_3]
   


    def forward_chop(self, x, shave=12):
        batchsize = 64
        h, w = x.size()[-2:]
        patch_size = 48
        padsize = int(patch_size)
        shave = int(patch_size / 2)

        scale = self.scale

        h_cut = (h - padsize) % (int(shave / 2))
        w_cut = (w - padsize) % (int(shave / 2))

        x_unfold = torch.nn.functional.unfold(x, padsize, stride=int(shave / 2)).transpose(0, 2).contiguous()
        x_hw_cut = x[..., (h - padsize):, (w - padsize):]
        y_hw_cut, m_hw_cut_list = self.forward(x_hw_cut.cuda())

        x_h_cut = x[..., (h - padsize):, :]
        x_w_cut = x[..., :, (w - padsize):]

        y_h_cut, m_h_cut_list = self.cut_h(x_h_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)
        y_w_cut, m_w_cut_list = self.cut_w(x_w_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)


        x_h_top = x[..., :padsize, :]
        x_w_top = x[..., :, :padsize]


        y_h_top, m_h_top_list = self.cut_h(x_h_top, h, w, h_cut, w_cut, padsize, shave, scale, batchsize) 
        y_w_top, m_w_top_list = self.cut_w(x_w_top, h, w, h_cut, w_cut, padsize, shave, scale, batchsize) 

        x_unfold = x_unfold.view(x_unfold.size(0), -1, padsize, padsize)


        y_unfold = []
        m_unfold_1 = []
        m_unfold_2 = []
        m_unfold_3 = []
        x_range = x_unfold.size(0) // batchsize + (x_unfold.size(0) % batchsize != 0)
        x_unfold.cuda()
        for i in range(x_range):
            sr, mask = self.forward(x_unfold[i * batchsize:(i + 1) * batchsize, ...])
            y_unfold.append(sr)
            m_unfold_1.append(mask[0])
            m_unfold_2.append(mask[1])
            m_unfold_3.append(mask[2])

          

        y_unfold = torch.cat(y_unfold, dim=0)
        y = torch.nn.functional.fold(y_unfold.view(y_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
                                     ((h - h_cut) * scale, (w - w_cut) * scale), padsize * scale,
                                     stride=int(shave / 2 * scale))



        y[..., :padsize * scale, :] = y_h_top
        y[..., :, :padsize * scale] = y_w_top

        y_unfold = y_unfold[..., int(shave / 2 * scale):padsize * scale - int(shave / 2 * scale),
                   int(shave / 2 * scale):padsize * scale - int(shave / 2 * scale)].contiguous()
        y_inter = torch.nn.functional.fold(y_unfold.view(y_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
                                           ((h - h_cut - shave) * scale, (w - w_cut - shave) * scale),
                                           padsize * scale - shave * scale, stride=int(shave / 2 * scale))



        y_ones = torch.ones(y_inter.shape, dtype=y_inter.dtype).cuda()
        divisor_y = torch.nn.functional.fold(
            torch.nn.functional.unfold(y_ones, padsize * scale - shave * scale, stride=int(shave / 2 * scale)),
            ((h - h_cut - shave) * scale, (w - w_cut - shave) * scale), padsize * scale - shave * scale,
            stride=int(shave / 2 * scale))

        y_inter = y_inter / divisor_y

        y[..., int(shave / 2 * scale):(h - h_cut) * scale - int(shave / 2 * scale),
        int(shave / 2 * scale):(w - w_cut) * scale - int(shave / 2 * scale)] = y_inter

        y = torch.cat([y[..., :y.size(2) - int((padsize - h_cut) / 2 * scale), :],
                       y_h_cut[..., int((padsize - h_cut) / 2 * scale + 0.5):, :]], dim=2)



        y_w_cat = torch.cat([y_w_cut[..., :y_w_cut.size(2) - int((padsize - h_cut) / 2 * scale), :],
                             y_hw_cut[..., int((padsize - h_cut) / 2 * scale + 0.5):, :]], dim=2)



        y = torch.cat([y[..., :, :y.size(3) - int((padsize - w_cut) / 2 * scale)],
                       y_w_cat[..., :, int((padsize - w_cut) / 2 * scale + 0.5):]], dim=3)

        #------------m--------------
        def process_mask(m_unfold, m_h_top, m_w_top, m_h_cut, m_w_cut, m_hw_cut):
            m_unfold = torch.cat(m_unfold, dim=0)
            m = torch.nn.functional.fold(m_unfold.view(m_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
                                         ((h - h_cut), (w - w_cut)), padsize,
                                         stride=int(shave / 2))
            m[..., :padsize, :] = m_h_top
            m[..., :, :padsize] = m_w_top
            m_unfold = m_unfold[..., int(shave / 2):padsize - int(shave / 2),
                       int(shave / 2):padsize - int(shave / 2)].contiguous()
            m_inter = torch.nn.functional.fold(m_unfold.view(m_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
                                               ((h - h_cut - shave), (w - w_cut - shave)),
                                               padsize - shave, stride=int(shave / 2))
            m_ones = torch.ones(m_inter.shape, dtype=m_inter.dtype).cuda()
            divisor_m = torch.nn.functional.fold(
                torch.nn.functional.unfold(m_ones, padsize - shave, stride=int(shave / 2)),
                ((h - h_cut - shave), (w - w_cut - shave)), padsize - shave,
                stride=int(shave / 2))
            m_inter = m_inter / divisor_m
            m[..., int(shave / 2):(h - h_cut) - int(shave / 2),
            int(shave / 2):(w - w_cut) - int(shave / 2)] = m_inter
            m = torch.cat([m[..., :m.size(2) - int((padsize - h_cut) / 2), :],
                           m_h_cut[..., int((padsize - h_cut) / 2 + 0.5):, :]], dim=2)
            m_w_cat = torch.cat([m_w_cut[..., :m_w_cut.size(2) - int((padsize - h_cut) / 2), :],
                                 m_hw_cut[..., int((padsize - h_cut) / 2 + 0.5):, :]], dim=2)

            m = torch.cat([m[..., :, :m.size(3) - int((padsize - w_cut) / 2)],
                           m_w_cat[..., :, int((padsize - w_cut) / 2 + 0.5):]], dim=3)
            return m

        m_1 = process_mask(m_unfold_1, m_h_top_list[0], m_w_top_list[0], m_h_cut_list[0], m_w_cut_list[0], m_hw_cut_list[0])
        m_2 = process_mask(m_unfold_2, m_h_top_list[1], m_w_top_list[1], m_h_cut_list[1], m_w_cut_list[1], m_hw_cut_list[1])
        m_3 = process_mask(m_unfold_3, m_h_top_list[2], m_w_top_list[2], m_h_cut_list[2], m_w_cut_list[2], m_hw_cut_list[2])

        return y.cuda(), [m_1.cuda(), m_2.cuda(), m_3.cuda()]

    def cut_h(self, x_h_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize):

        x_h_cut_unfold = torch.nn.functional.unfold(x_h_cut, padsize, stride=int(shave / 2)).transpose(0, 2).contiguous()
        x_h_cut_unfold = x_h_cut_unfold.view(x_h_cut_unfold.size(0), -1, padsize, padsize)
    

        x_range = x_h_cut_unfold.size(0) // batchsize + (x_h_cut_unfold.size(0) % batchsize != 0)
        y_h_cut_unfold = []
        m_h_cut_unfold_1 = []
        m_h_cut_unfold_2 = []
        m_h_cut_unfold_3 = []
        x_h_cut_unfold.cuda()
        for i in range(x_range):
           
            sr, mask = self.forward(x_h_cut_unfold[i * batchsize:(i + 1) * batchsize, ...])
            y_h_cut_unfold.append(sr)
            m_h_cut_unfold_1.append(mask[0])
            m_h_cut_unfold_2.append(mask[1])
            m_h_cut_unfold_3.append(mask[2])
           
        y_h_cut_unfold = torch.cat(y_h_cut_unfold, dim=0)
        y_h_cut = torch.nn.functional.fold(
            y_h_cut_unfold.view(y_h_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
            (padsize * scale, (w - w_cut) * scale), padsize * scale, stride=int(shave / 2 * scale))
        y_h_cut_unfold = y_h_cut_unfold[..., :,
                         int(shave / 2 * scale):padsize * scale - int(shave / 2 * scale)].contiguous()
        y_h_cut_inter = torch.nn.functional.fold(
            y_h_cut_unfold.view(y_h_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
            (padsize * scale, (w - w_cut - shave) * scale), (padsize * scale, padsize * scale - shave * scale),
            stride=int(shave / 2 * scale))

        y_ones = torch.ones(y_h_cut_inter.shape, dtype=y_h_cut_inter.dtype).cuda()
        divisor = torch.nn.functional.fold(
            torch.nn.functional.unfold(y_ones, (padsize * scale, padsize * scale - shave * scale),
                                       stride=int(shave / 2 * scale)), (padsize * scale, (w - w_cut - shave) * scale),
            (padsize * scale, padsize * scale - shave * scale), stride=int(shave / 2 * scale))
        y_h_cut_inter = y_h_cut_inter / divisor

        y_h_cut[..., :, int(shave / 2 * scale):(w - w_cut) * scale - int(shave / 2 * scale)] = y_h_cut_inter

        #-----------
        def process_m(m_h_cut_unfold):
            m_h_cut_unfold = torch.cat(m_h_cut_unfold, dim=0)
            m_h_cut = torch.nn.functional.fold(
                m_h_cut_unfold.view(m_h_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
                (padsize, (w - w_cut)), padsize, stride=int(shave / 2 ))

            m_h_cut_unfold = m_h_cut_unfold[..., :, int(shave / 2):padsize - int(shave / 2)].contiguous()
            m_h_cut_inter = torch.nn.functional.fold(
                m_h_cut_unfold.view(m_h_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
                (padsize, (w - w_cut - shave)), (padsize, padsize - shave),
                stride=int(shave / 2))

            m_ones = torch.ones(m_h_cut_inter.shape, dtype=m_h_cut_inter.dtype).cuda()
            divisor = torch.nn.functional.fold(
                torch.nn.functional.unfold(m_ones, (padsize, padsize - shave),
                                           stride=int(shave / 2)), (padsize, (w - w_cut - shave)),
                (padsize, padsize - shave), stride=int(shave / 2))

            m_h_cut_inter = m_h_cut_inter / divisor

            m_h_cut[..., :, int(shave / 2):(w - w_cut) - int(shave / 2)] = m_h_cut_inter
            return m_h_cut

        m_h_cut_1 = process_m(m_h_cut_unfold_1)
        m_h_cut_2 = process_m(m_h_cut_unfold_2)
        m_h_cut_3 = process_m(m_h_cut_unfold_3)

        return y_h_cut, [m_h_cut_1, m_h_cut_2, m_h_cut_3]

    def cut_w(self, x_w_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize): 

        x_w_cut_unfold = torch.nn.functional.unfold(x_w_cut, padsize, stride=int(shave / 2)).transpose(0,
                                                                                                       2).contiguous()
      

        x_w_cut_unfold = x_w_cut_unfold.view(x_w_cut_unfold.size(0), -1, padsize, padsize)
      

        x_range = x_w_cut_unfold.size(0) // batchsize + (x_w_cut_unfold.size(0) % batchsize != 0)
        y_w_cut_unfold = []
        m_w_cut_unfold_1 = []
        m_w_cut_unfold_2 = []
        m_w_cut_unfold_3 = []
        x_w_cut_unfold.cuda()
        for i in range(x_range):
           
            sr, mask = self.forward(x_w_cut_unfold[i * batchsize:(i + 1) * batchsize, ...])
           
            y_w_cut_unfold.append(sr)
            m_w_cut_unfold_1.append(mask[0])
            m_w_cut_unfold_2.append(mask[1])
            m_w_cut_unfold_3.append(mask[2])
           
        y_w_cut_unfold = torch.cat(y_w_cut_unfold, dim=0)

        y_w_cut = torch.nn.functional.fold(
            y_w_cut_unfold.view(y_w_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
            ((h - h_cut) * scale, padsize * scale), padsize * scale, stride=int(shave / 2 * scale))
        y_w_cut_unfold = y_w_cut_unfold[..., int(shave / 2 * scale):padsize * scale - int(shave / 2 * scale),
                         :].contiguous()
        y_w_cut_inter = torch.nn.functional.fold(
            y_w_cut_unfold.view(y_w_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
            ((h - h_cut - shave) * scale, padsize * scale), (padsize * scale - shave * scale, padsize * scale),
            stride=int(shave / 2 * scale))

        y_ones = torch.ones(y_w_cut_inter.shape, dtype=y_w_cut_inter.dtype).cuda()
        divisor = torch.nn.functional.fold(
            torch.nn.functional.unfold(y_ones, (padsize * scale - shave * scale, padsize * scale),
                                       stride=int(shave / 2 * scale)), ((h - h_cut - shave) * scale, padsize * scale),
            (padsize * scale - shave * scale, padsize * scale), stride=int(shave / 2 * scale))
        y_w_cut_inter = y_w_cut_inter / divisor

        y_w_cut[..., int(shave / 2 * scale):(h - h_cut) * scale - int(shave / 2 * scale), :] = y_w_cut_inter

        def process_m(m_w_cut_unfold):
            m_w_cut_unfold = torch.cat(m_w_cut_unfold, dim=0)
            m_w_cut = torch.nn.functional.fold(
                m_w_cut_unfold.view(m_w_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
                ((h - h_cut), padsize), padsize, stride=int(shave / 2))
            m_w_cut_unfold = m_w_cut_unfold[..., int(shave / 2):padsize - int(shave / 2),
                             :].contiguous()
            m_w_cut_inter = torch.nn.functional.fold(
                m_w_cut_unfold.view(m_w_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
                ((h - h_cut - shave), padsize), (padsize - shave, padsize),
                stride=int(shave / 2))

            m_ones = torch.ones(m_w_cut_inter.shape, dtype=m_w_cut_inter.dtype).cuda()
            divisor = torch.nn.functional.fold(
                torch.nn.functional.unfold(m_ones, (padsize - shave, padsize),
                                           stride=int(shave / 2)), ((h - h_cut - shave), padsize),
                (padsize - shave, padsize), stride=int(shave / 2))

            m_w_cut_inter = m_w_cut_inter / divisor

            m_w_cut[..., int(shave / 2):(h - h_cut) - int(shave / 2), :] = m_w_cut_inter
            return m_w_cut

        m_w_cut_1 = process_m(m_w_cut_unfold_1)
        m_w_cut_2 = process_m(m_w_cut_unfold_2)
        m_w_cut_3 = process_m(m_w_cut_unfold_3)

        return y_w_cut, [m_w_cut_1, m_w_cut_2, m_w_cut_3]


    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


