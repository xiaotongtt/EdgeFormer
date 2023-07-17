import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import torch

import utility
import data
import model
import loss
from option import args
from importlib import import_module
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    global model
    if checkpoint.ok:
        # args.model = 'EDSR_REFINETOR'
        # args.save = 'edsr_refinetor'
        loader = data.Data(args)
        _model = model.Model(args, checkpoint)
        # print('Total params: %.2fM' % (sum(p.numel() for p in _model.parameters())/1000000.0))
        print('Total params: %fM' % (sum(p.numel() for p in _model.parameters())))
        _loss = loss.Loss(args, checkpoint) if not args.test_only else None

        module = import_module('trainer.' + args.trainer.lower())
        t = module.make_trainer(args, loader, _model, _loss, checkpoint)

        while not t.terminate():
            t.train()
            t.test()

        checkpoint.done()


if __name__ == '__main__':
    main()
