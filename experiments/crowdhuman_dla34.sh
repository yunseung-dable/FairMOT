cd src
python train.py mot --exp_id crowdhuman_dla34 --gpus 0,1,2,3  --batch_size 8 --num_iters -1 --resume --num_epochs 60 --load_model '/data/models/mot/fairmot/crowdhuman_dla34/model_25.pth'  --lr_step '50' --data_cfg '../src/lib/cfg/crowdhuman.json'
cd ..
