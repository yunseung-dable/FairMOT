cd src
python train.py mot --exp_id david893_dla34 --arch dla_34 --gpus 0,1,2,3  --batch_size 8 --num_iters -1  --num_epochs 30 --load_model '/data/vv-FairMOT/weights/ctdet_coco_dla_2x.pth' --lr_step '20' --save_all --data_cfg '../src/lib/cfg/david_893.json'
cd ..
