#!/bin/bash

cd /mnt/fast/nobackup/users/ds01502/MajorProjectCustomizableConditioningGANS/GANSketching


/mnt/fast/nobackup/users/ds01502/miniconda3/envs/gantrain/bin/python train.py \
--name sketch_facen_1_image --size 1024 --batch 1 \
--dataroot_sketch ./data/sketch/by_author/facen_1 \
--dataroot_image ./data/image/faces --l_image 0.7 \
--g_pretrained ./pretrained/stylegan2-ffhq/netG.pth \
--d_pretrained ./pretrained/stylegan2-ffhq/netD.pth \
--transform_fake down2,toSketch,up2,to3ch --transform_real down2,up2,to3ch \
--disable_eval --reduce_visuals --diffaug_policy translation --save_freq 125000 \
--max_iter 126000 \
