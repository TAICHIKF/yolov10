mv /mnt/WD_44T/kongfei/data/*   /mnt/data7T/kongfeidata/data/
mv /mnt/WD_44T/kongfei/coco/*.txt  /xmnt/mnt_nfs_qynas_v5/kongfei/coco/
cp -r /xmnt/mnt_nfs_qynas_v5/kongfei/coco  /mnt/data7T/kongfeidata/data
scp -r /Users/feikong/Downloads/data/  kongfei@172.22.162.127:/mnt/data7T/kongfeidata

ssh-keygen -t rsa -b 4096 -C "1282328191@qq.com"
cat ~/.ssh/id_rsa.pub
# add ssh key
ssh -T git@github.com

# second 
git clone git@github.com:TAICHIKF/yolov10.git


---
merge:
git status
git fetch origin
git log HEAD..origin/main
git config pull.rebase false
git pull origin main


conda create -n yolo python=3.9 -y
conda activate yolo
pip install -r requirements.txt
pip install -e .

pip install ultralytics==8.1.34 -i https://pypi.tuna.tsinghua.edu.cn/simple 
pip install openai -i https://pypi.tuna.tsinghua.edu.cn/simple 
pip install zhipuai -i https://pypi.tuna.tsinghua.edu.cn/simple 
pip install wandb -i https://pypi.tuna.tsinghua.edu.cn/simple 
pip install torchsummary -i https://pypi.tuna.tsinghua.edu.cn/simple 