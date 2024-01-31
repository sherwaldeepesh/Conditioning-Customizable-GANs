#!/bin/bash
cd /mnt/fast/nobackup/users/ds01502/MajorProjectCustomizableConditioningGANS/GANSketching


/mnt/fast/nobackup/users/ds01502/miniconda3/envs/gantrain/bin/python train.py \
--name photosketch_horse_riders_30 \
--dataroot_sketch ./data/sketch/photosketch/horse_riders \
--dataroot_image /mnt/fast/nobackup/scratch4weeks/ds01502/Data/dataFolder/data/image/horse_n --l_image 0.5 \
--eval_dir ./data/eval/horse_riders \
--g_pretrained ./pretrained/stylegan2-horse/netG.pth \
--d_pretrained ./pretrained/stylegan2-horse/netD.pth \
--disable_eval --max_iter 150000 --diffaug_policy translation \
--save_freq 145000 \


/mnt/fast/nobackup/users/ds01502/miniconda3/envs/gantrain/bin/python train.py \
--name photosketch_horse_riders_1 \
--dataroot_sketch ./data/sketch/photosketch/horse_riders_1 \
--dataroot_image /mnt/fast/nobackup/scratch4weeks/ds01502/Data/dataFolder/data/image/horse_n --l_image 0.5 \
--eval_dir ./data/eval/horse_riders \
--g_pretrained ./pretrained/stylegan2-horse/netG.pth \
--d_pretrained ./pretrained/stylegan2-horse/netD.pth \
--disable_eval --max_iter 150000 --diffaug_policy translation \
--save_freq 145000 \


/mnt/fast/nobackup/users/ds01502/miniconda3/envs/gantrain/bin/python train.py \
--name photosketch_horse_riders_5 \
--dataroot_sketch ./data/sketch/photosketch/horse_riders_5 \
--dataroot_image /mnt/fast/nobackup/scratch4weeks/ds01502/Data/dataFolder/data/image/horse_n --l_image 0.5 \
--eval_dir ./data/eval/horse_riders \
--g_pretrained ./pretrained/stylegan2-horse/netG.pth \
--d_pretrained ./pretrained/stylegan2-horse/netD.pth \
--disable_eval --max_iter 150000 --diffaug_policy translation \
--save_freq 145000 \
