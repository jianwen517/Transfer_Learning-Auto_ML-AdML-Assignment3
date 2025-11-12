#!/bin/bash
#$ -M jwu35@nd.edu                 
#$ -m abe                         
#$ -pe smp 8                       
#$ -q gpu                         
#$ -l gpu_card=1                   
#$ -N transfer_autoML            
#$ -cwd                           
#$ -o /users/jwu35/Myspace/Project/NeSy/Ad/logs/transfer_autoML.log  # 输出日志路径


echo "[$(date)] Job started on $(hostname)"


source ~/.bashrc
conda activate transfer
echo "Python executable: $(which python)"
python -c "import torch; print('Torch:', torch.__version__, 'CUDA available:', torch.cuda.is_available(), 'Device:', torch.cuda.get_device_name(0))"


echo "[$(date)] Starting AutoML training..."
python Transfer.py

echo "[$(date)] Job finished."
