cd /mnt/fast/nobackup/users/ds01502/MajorProjectCustomizableConditioningGANS/GANSketching


/mnt/fast/nobackup/users/ds01502/miniconda3/envs/tf114/bin/python run_metrics.py --models_list weights/eval_list --output metric_results.csv

#/mnt/fast/nobackup/users/ds01502/miniconda3/envs/gantrain/bin/python generate.py --ckpt checkpoint/sketch_facen_1_image/125000_net_G.pth --save_dir output/sketch_facen_1_image --size 1024 --batch_size 20