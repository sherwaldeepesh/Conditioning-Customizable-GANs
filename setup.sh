wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p miniconda3
rm Miniconda3-latest-Linux-x86_64.sh
conda=miniconda3/bin/conda
$conda create -y -n gansk python=3.7 pip 
$conda install -n gansk -y pytorch torchvision cudatoolkit=10.1 -c pytorch
$pip install lpips clean-fid pandas
$conda install -n gansk tensorflow-gpu=1.14.0
