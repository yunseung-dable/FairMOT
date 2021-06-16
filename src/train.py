from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import json
import torch
import torch.utils.data
from torchvision.transforms import transforms as T
from opts import opts
from models.model import create_model, load_model, save_model
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
import warnings
import gc
# warnings.filterwarnings("ignore", category=UserWarning)

def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    print('Setting up data...')
    Dataset = get_dataset(opt.dataset, opt.task)

    f = open(opt.data_cfg)
    data_config = json.load(f)
    trainset_paths = data_config['train']
    validset_paths = data_config['valid']
    dataset_root = data_config['root']
    f.close()
    transforms = T.Compose([T.ToTensor()])
    dataset_train = Dataset(opt, dataset_root, trainset_paths, (1088, 608), augment=True, transforms=transforms)
    # dataset_train = Dataset(opt, dataset_root, trainset_paths, (1088, 608), augment=False, transforms=transforms)
    # dataset_val = Dataset(opt, dataset_root, validset_paths, (1088, 608), augment=False, transforms=transforms)
    opt = opts().update_dataset_info_and_set_heads(opt, dataset_train)
    print(opt)

    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0

    # Get dataloader

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # val_loader = torch.utils.data.DataLoader(
    #     dataset_val,
    #     batch_size=opt.batch_size,
    #     shuffle=True,
    #     num_workers=opt.num_workers,
    #     pin_memory=True,
    #     drop_last=True
    # )

    print('Starting training...')
    Trainer = train_factory[opt.task]
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, trainer.optimizer, opt.resume, opt.lr, opt.lr_step)


    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('train_{} {:5f} | '.format(k, round(v,3)))

        torch.cuda.empty_cache()
        gc.collect()
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                       epoch, model, optimizer)
            # log_dict_val, _ = trainer.valid(epoch, val_loader)
            # for k, v in log_dict_val.items():
            #     logger.scalar_summary('test_{}'.format(k), v, epoch)
            #     logger.write('val_{} {:5f} | '.format(k, round(v,3)))

        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                       epoch, model, optimizer)
        logger.write('\n')
        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if epoch % 5 == 0 or epoch >= 25:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)

    logger.close()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    opt = opts().parse()
    main(opt)
