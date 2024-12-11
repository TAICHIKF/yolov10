mv /mnt/WD_44T/kongfei/data/*   /mnt/data7T/kongfeidata/data/
mv /mnt/WD_44T/kongfei/coco/*.txt  /xmnt/mnt_nfs_qynas_v5/kongfei/coco/


cp -r /xmnt/mnt_nfs_qynas_v5/kongfei/coco  /mnt/data7T/kongfeidata/data

scp -r /Users/feikong/Downloads/data/  kongfei@172.22.162.127:/mnt/data7T/kongfeidata



pip install openai -i https://pypi.tuna.tsinghua.edu.cn/simple 
pip install zhipuai -i https://pypi.tuna.tsinghua.edu.cn/simple 
pip install wandb -i https://pypi.tuna.tsinghua.edu.cn/simple 
pip install torchsummary -i https://pypi.tuna.tsinghua.edu.cn/simple 