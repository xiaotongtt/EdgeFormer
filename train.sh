#!/bin/bash
# python main.py --n_GPUs 1 --model edgeformer --loss '1*L1+1*ES' --save edgeformer_x4_no_sobel --scale 4  --trainer trainer_edgeformer --epochs 1000 --n_resblocks 32 --refinetor_deep 3 --batch_size 16 --n_feats 128 --decay 200-400-600-800
# python main.py --n_GPUs 1 --model edgeformer --loss '1*L1+1*ES' --save edgeformer_x4_sobel_v2 --scale 4  --trainer trainer_edgeformer --epochs 1000 --n_resblocks 32 --refinetor_deep 3 --batch_size 16 --n_feats 128 --decay 200-400-600-800
python main.py --n_GPUs 1 --model edgeformer --loss '1*L1+1*ES' --save edgeformer_x3_sobel_v2 --scale 3 --patch_size 144  --trainer trainer_edgeformer --epochs 1000 --n_resblocks 32 --refinetor_deep 3 --batch_size 16 --n_feats 128 --decay 200-400-600-800