from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import logging
import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track import eval_seq
import time


logger.setLevel(logging.INFO)


def demo(opt):
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    logger.info('Starting tracking...')
    dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
    result_filename = os.path.join(result_root, 'results.txt')
    frame_rate = dataloader.frame_rate

    frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')
    eval_seq(opt, dataloader, 'mot', result_filename,
             save_dir=frame_dir, show_image=False, frame_rate=frame_rate,
             use_cuda=opt.gpus!=[-1])

    if opt.output_format == 'video':
        _, mon, day,hour,min,_,_,_,_ = time.localtime()
        mon = '0' + str(mon) if mon < 10 else str(mon)
        day = '0' + str(day) if day < 10 else str(day)
        hour = '0' + str(hour) if hour < 10 else str(hour)
        min = '0' + str(min) if min < 10 else str(min)

        output_video_path = osp.join(result_root, mon+day+hour+min)
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(osp.join(result_root, 'frame'), 'test_'+output_video_path+'.mp4')
        os.system('rm -rf /data/models/mot/test/demo_results.mp4')
        os.system(cmd_str)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    print(f'using GPU {opt.gpus} ')
    demo(opt)
