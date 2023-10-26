# cd /workspace/pytorch-segmentation
git config --global user.email "susannju@163.com"
git config --global user.name "susanbao"
git remote set-url origin https://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@github.com/susanbao/pytorch-segmentation.git

pip install ipdb wandb Cython scipy future scikit-learn scikit-image requests pandas matplotlib seaborn ml-collections packaging
apt-get install libsm6 libxrender1 libfontconfig1 libxext6
pip install -r requirements.txt
apt-get install zip

export LC_ALL=C.UTF-8
export LANG=C.UTF-8
wandb login 0fdb5bef26b98a4c93e80ff72f9a0121d0391ae8

cd apex
python setup.py install