#!/bin/bash
cd /mnt/fast/nobackup/users/ds01502/MajorProjectCustomizableConditioningGANS/GANSketching


/mnt/fast/nobackup/users/ds01502/miniconda3/envs/gantrain/bin/python train.py \
--name teasercat_lsketch --batch 4 \
--dataroot_sketch ./data/sketch/by_author/cat \
--dataroot_image ./data/image/cat --l_image 0.5 \
--g_pretrained ./pretrained/stylegan2-cat/netG.pth \
--disable_eval --max_iter 126000 --diffaug_policy translation \
--save_freq 125000 \

# /mnt/fast/nobackup/users/ds01502/miniconda3/envs/gantrain/bin/python train.py \
# --name teasercat_lsketch_aug --batch 1 \
# --dataroot_sketch ./data/sketch/by_author/cat \
# --dataroot_image ./data/image/cat --l_image 0.5 \
# --g_pretrained ./pretrained/stylegan2-cat/netG.pth \
# --transform_fake down2,toSketch,up2,to3ch --transform_real down2,up2,to3ch \
# --disable_eval --max_iter 126000 --diffaug_policy translation \
# --save_freq 125000 \
  

# /mnt/fast/nobackup/users/ds01502/miniconda3/envs/gantrain/bin/python train.py \
# --name teasercat_lsketch_limage --batch 1 \
# --dataroot_sketch ./data/sketch/by_author/cat \
# --dataroot_image ./data/image/cat --l_image 0.5 \
# --g_pretrained ./pretrained/stylegan2-cat/netG.pth \
# --d_pretrained ./pretrained/stylegan2-cat/netD.pth \
# --disable_eval --max_iter 126000 --diffaug_policy translation \
# --save_freq 125000 \

# /mnt/fast/nobackup/users/ds01502/miniconda3/envs/gantrain/bin/python train.py \
# --name teasercat_lsketch_limage_aug --batch 1 \
# --dataroot_sketch ./data/sketch/by_author/cat \
# --dataroot_image ./data/image/cat --l_image 0.5 \
# --g_pretrained ./pretrained/stylegan2-cat/netG.pth \
# --d_pretrained ./pretrained/stylegan2-cat/netD.pth \
# --disable_eval --max_iter 126000 --diffaug_policy translation \
# --transform_fake down2,toSketch,up2,to3ch --transform_real down2,up2,to3ch \
# --save_freq 125000 \