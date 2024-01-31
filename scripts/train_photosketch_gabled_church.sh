#!/bin/bash
cd /mnt/fast/nobackup/users/ds01502/MajorProjectCustomizableConditioningGANS/GANSketching


/mnt/fast/nobackup/users/ds01502/miniconda3/envs/gantrain/bin/python train.py \
--name photosketch_gabled_church \
--dataroot_sketch ./data/sketch/photosketch/gabled_church \
--dataroot_image /mnt/fast/nobackup/scratch4weeks/ds01502/Data/dataFolder/data/image/church --l_image 0.5 \
--eval_dir ./data/eval/gabled_church \
--g_pretrained ./pretrained/stylegan2-church/netG.pth \
--d_pretrained ./pretrained/stylegan2-church/netD.pth \
--disable_eval --max_iter 126000 --diffaug_policy translation \
--save_freq 125000 \


/mnt/fast/nobackup/users/ds01502/miniconda3/envs/gantrain/bin/python train.py \
--name photosketch_gabled_church \
--dataroot_sketch ./data/sketch/photosketch/gabled_church_1 \
--dataroot_image /mnt/fast/nobackup/scratch4weeks/ds01502/Data/dataFolder/data/image/church --l_image 0.5 \
--eval_dir ./data/eval/gabled_church \
--g_pretrained ./pretrained/stylegan2-church/netG.pth \
--d_pretrained ./pretrained/stylegan2-church/netD.pth \
--disable_eval --max_iter 126000 --diffaug_policy translation \
--save_freq 125000 \


/mnt/fast/nobackup/users/ds01502/miniconda3/envs/gantrain/bin/python train.py \
--name photosketch_gabled_church \
--dataroot_sketch ./data/sketch/photosketch/gabled_church_5 \
--dataroot_image /mnt/fast/nobackup/scratch4weeks/ds01502/Data/dataFolder/data/image/church --l_image 0.5 \
--eval_dir ./data/eval/gabled_church \
--g_pretrained ./pretrained/stylegan2-church/netG.pth \
--d_pretrained ./pretrained/stylegan2-church/netD.pth \
--disable_eval --max_iter 126000 --diffaug_policy translation \
--save_freq 125000 \
