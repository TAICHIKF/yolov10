mv /mnt/WD_44T/kongfei/data/*   /mnt/data7T/kongfeidata/data/
mv /mnt/WD_44T/kongfei/coco/*.txt  /xmnt/mnt_nfs_qynas_v5/kongfei/coco/
cp -r /xmnt/mnt_nfs_qynas_v5/kongfei/coco  /mnt/data7T/kongfeidata/data
scp -r /home/kongfei/code/yolov10/train_val/cfg_llm/model/Poe_10  kongfei@172.22.162.38:/home/kongfei/code/yolov10/train_val/cfg_llm/model


scp -r  /home/kongfei/code/yolov10/train_val/cfg_llm/models_new/v8_Poe_30_m  kongfei@172.22.162.38:/home/kongfei/code/yolov10/train_val/cfg_llm/models_new
scp -r  /home/kongfei/code/yolov10/train_val/cfg_llm/models_new/v8_Poe_30_x  kongfei@172.22.162.38:/home/kongfei/code/yolov10/train_val/cfg_llm/models_new

ssh-keygen -t rsa -b 4096 -C "1282328191@qq.com"
cat ~/.ssh/id_rsa.pub
# add ssh key
ssh -T git@github.com

# second 
git clone git@github.com:TAICHIKF/yolov10.git


# data to u209:
# scp -r /xmnt/mnt_nfs_qynas_v4/kongfei/data/coco/images/train2017  kongfei@172.18.4.209:/mnt/mnt_dmdisk_000_20T/kongfei/data/coco/images   
# scp -r /xmnt/mnt_nfs_qynas_v4/kongfei/data/coco/images/val2017  kongfei@172.18.4.209:/mnt/mnt_dmdisk_000_20T/kongfei/data/coco/images   
# scp -r /xmnt/mnt_nfs_qynas_v4/kongfei/data/coco/annotations  kongfei@172.18.4.209:/mnt/mnt_dmdisk_000_20T/kongfei/data/coco   
# scp -r /xmnt/mnt_nfs_qynas_v4/kongfei/data/coco/*.txt  kongfei@172.18.4.209:/mnt/mnt_dmdisk_000_20T/kongfei/data/coco  




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