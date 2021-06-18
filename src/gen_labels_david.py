import os.path as osp
import os
import cv2
import json

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


def gen_ann_list(file_list, ann_root):
    files = []
    with open(file_list, 'r') as f:
	    for file in f.readlines():
		    files.append( os.path.join(ann_root, file.strip()+'.json') )
    return files

def write_to_text(tid_curr: int, 
                  img_width: int, 
                  img_height: int, 
                  labels: dict, 
                  label_fpath: str):
    """
    labels: 
    {
        1: {
            'head_box': { 'x': 862, 'y': 94,  'width': 15, 'height': 58 }, 
            'visible_box': { 'x': 866, 'y': 93, 'width': 10, 'height': 13 },
            },
        ...
    }
    """
    
    for label in labels: # for each object
        hbox_x = label['head_box']['x']
        hbox_y = label['head_box']['y']
        hbox_w = label['head_box']['width']
        hbox_h = label['head_box']['height']

        hbox_x += hbox_w / 2 # center x
        hbox_y += hbox_h / 2 # center y

        vbox_x = label['visible_box']['x']
        vbox_y = label['visible_box']['y']
        vbox_w = label['visible_box']['width']
        vbox_h = label['visible_box']['height']

        vbox_x += vbox_w / 2 # center x
        vbox_y += vbox_h / 2 # center y

        label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                tid_curr, hbox_x / img_width, hbox_y / img_height, hbox_w / img_width, hbox_h / img_height, 
                vbox_x / img_width, vbox_y / img_height, vbox_w / img_width, vbox_h / img_height)

        with open(label_fpath, 'a') as f:
            f.write(label)

def gen_labels_david(file_list, label_root, ann_root):
    """
    file_list: DAVID_893/*.txt
    data_root: DAVID_893/images/
    label_root: DAVID_893/labels_with_id/ -> 생성
    ann_root: DAVID_893/jsons
    """
    mkdirs(label_root)

    # generate train or val list 
    anns_data = gen_ann_list(file_list, ann_root)

    tid_curr = 0
    for i, ann_data in enumerate(anns_data): # each json file
        # read image
        img_path = ann_data.replace('jsons', 'images').replace('.json', '.jpg')
        img = cv2.imread(
            img_path,
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        img_height, img_width = img.shape[0:2]

        # open json
        with open(ann_data, 'r') as ann_json:
            anns = json.load(ann_json)

        # touch label-file
        label_fpath = os.path.join(label_root, os.path.basename(img_path).replace('.jpg', '.txt'))
        f = open(label_fpath, 'w')
        f.close()

        # empty check
        if len(anns) == 0:
            continue
        
        # label dictionary per an image
        labels = dict()

        for j in range(len(anns)): # each object(head or visible)
            person_id = anns[j]['classification']['attributes'][0]['value'] if anns[j]['classification']['attributes'][0]['code'] == 'id' else anns[j]['classification']['attributes'][1]['value'] 
            person_id = int(person_id)
            mode = anns[j]['classification']['code']

            if person_id not in labels.keys():
                labels[person_id] = dict([(key, dict()) for key in ['head_box', 'visible_box']])
            
            if mode == 'head':
                labels[person_id]['head_box'] = anns[j]['label']['data']
            elif mode  == 'visible':
                labels[person_id]['visible_box'] = anns[j]['label']['data']

        write_to_text(labels)

if __name__ == '__main__':
    ann_root = '/mnt/sda1/user/data/DAVID_893/jsons'
    label_root_train = '/mnt/sda1/user/data/DAVID_893/labels_with_ids/train'
    label_root_val = '/mnt/sda1/user/data/DAVID_893/labels_with_ids/val'
    file_list_train = '/mnt/sda1/user/data/DAVID_893/train.txt'
    file_list_val = '/mnt/sda1/user/data/DAVID_893/val.txt'
    
    gen_labels_david(file_list_train, label_root_train, ann_root)
    gen_labels_david(file_list_val, label_root_val, ann_root)