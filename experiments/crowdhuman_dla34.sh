cd src
python train.py mot --exp_id crowdhuman_dla34 --gpus -1  --batch_size 1 --num_epochs 60 --lr_step '50' --data_cfg '../src/lib/cfg/crowdhuman.json'
cd ..
