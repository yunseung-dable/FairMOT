import os.path as osp
import os
import cv2
import json
import numpy as np


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


def load_func(fpath):
    print('fpath', fpath)
    assert os.path.exists(fpath)
    with open(fpath, 'r') as fid:
        lines = fid.readlines()
    records =[json.loads(line.strip('\n')) for line in lines]
    return records


def gen_labels_crowd(data_root, label_root, ann_root):
    mkdirs(label_root)
    anns_data = load_func(ann_root)

    tid_curr = 0
    for i, ann_data in enumerate(anns_data):
        print(i)
        image_name = '{}.jpg'.format(ann_data['ID'])
        img_path = os.path.join(data_root, image_name)
        anns = ann_data['gtboxes']
        img = cv2.imread(
            img_path,
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        img_height, img_width = img.shape[0:2]
        for i in range(len(anns)):
            if 'extra' in anns[i] and 'ignore' in anns[i]['extra'] and anns[i]['extra']['ignore'] == 1:
                continue
            head_x, head_y, head_w, head_h = anns[i]['hbox']
            head_x += head_w / 2
            head_y += head_h / 2

            full_x, full_y, full_w, full_h = anns[i]['fbox']
            full_x += full_w / 2
            full_y += full_h / 2

            label_fpath = img_path.replace('images', 'labels_with_ids_both').replace('.png', '.txt').replace('.jpg', '.txt')
            label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                tid_curr, head_x / img_width, head_y / img_height, head_w / img_width, head_h / img_height,
                full_x / img_width, full_y / img_height, full_w / img_width, full_h / img_height)
            with open(label_fpath, 'a') as f:
                f.write(label_str)
            tid_curr += 1


if __name__ == '__main__':
    data_val = '/data/mot/crowdhuman/images/val'
    label_val = '/data/mot/crowdhuman/labels_with_ids_both/val'
    ann_val = '/data/mot/crowdhuman/annotation_val.odgt'
    data_train = '/data/mot/crowdhuman/images/train'
    label_train = '/data/mot/crowdhuman/labels_with_ids_both/train'
    ann_train = '/data/mot/crowdhuman/annotation_train.odgt'
    gen_labels_crowd(data_train, label_train, ann_train)
    gen_labels_crowd(data_val, label_val, ann_val)


