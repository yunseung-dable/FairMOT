cd src
python train.py mot --exp_id crowdhuman_dla34 --gpus 1,2,3  --batch_size 8 --num_epochs 60 --load_model '../exp/mot/crowdhuman_dla34/model_last.pth' --lr_step '50' --data_cfg '../src/lib/cfg/crowdhuman.json'
cd ..
