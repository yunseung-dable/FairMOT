import os.path as osp
import os
import cv2
import json

def mkdirs(path):
    os.makedirs(path, exist_ok=True)


def create_annot_file_list(file_path_txt, annot_root):
	annot_file_path_list = []
	with open(file_path_txt, 'r') as f:
		for annot_file_path in f.readlines():
			annot_file_path_list.append( os.path.join(annot_root, f"{annot_file_path.strip()}.json") )
	return annot_file_path_list


def process_nonprovide_box(label):
    fill_dict = {'x': -1, 'y': -1, 'width': -1, 'height': -1}
    if label['head'] == {}:
        label['head'] = fill_dict

    elif label['visible'] == {}:
        label['visible'] = fill_dict

    return label


def write_to_text(img_width: int, 
                  img_height: int, 
                  labels: dict, 
                  label_fpath: str):
    """
    labels: 
    {
        '1': {
            'head': { 'x': 862, 'y': 94,  'width': 15, 'height': 58 }, 
            'visible': { 'x': 866, 'y': 93, 'width': 10, 'height': 13 },
            'overall_person_id': 1
            },
        ...
    }
    """
    
    for i in labels.keys(): # for each object
        labels[i] = process_nonprovide_box(labels[i])

        overall_person_id = labels[i]['overall_person_id']
        
        hbox_x = labels[i]['head']['x']
        hbox_y = labels[i]['head']['y']
        hbox_w = labels[i]['head']['width']
        hbox_h = labels[i]['head']['height']

        hbox_x += hbox_w / 2 # center x
        hbox_y += hbox_h / 2 # center y

        vbox_x = labels[i]['visible']['x']
        vbox_y = labels[i]['visible']['y']
        vbox_w = labels[i]['visible']['width']
        vbox_h = labels[i]['visible']['height']

        vbox_x += vbox_w / 2 # center x
        vbox_y += vbox_h / 2 # center y

        label_str = f'0 {overall_person_id :d} {hbox_x / img_width :.6f} {hbox_y / img_height :.6f} {hbox_w / img_width :.6f} {hbox_h / img_height :.6f} {vbox_x / img_width :.6f} {vbox_y / img_height :.6f} {vbox_w / img_width :.6f} {vbox_h / img_height :.6f}\n'

        with open(label_fpath, 'a') as f:
            f.write(label_str)


def gen_labels_david(file_list, label_root, ann_root):
    """
    file_list: DAVID_893/*.txt
    label_root: DAVID_893/labels_with_id_both/ -> 생성
    ann_root: DAVID_893/jsons
    """
    mkdirs(label_root)

    # generate train or val list 
    anns_data = create_annot_file_list(file_list, ann_root)

    overall_person_id = 0
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

        # for negative samples
        if len(anns) == 0:   
            f = open(label_fpath, 'w')
            f.close()
            continue
        
        # label dictionary per an image
        labels = dict()

        for ann in anns: # each object(head or visible)
            attributes = ann['classification']['attributes'] 
            if attributes[0]['code'] == 'id':
                person_id = attributes[0]['value']
            else:
               person_id = attributes[1]['value'] 
            mode = ann['classification']['code'] # head or visible

            if person_id not in labels.keys():
                labels[person_id] = dict([(key, dict()) for key in ['head', 'visible']])
                overall_person_id += 1 # 1부터 시작
            
            labels[person_id][mode] = ann['label']['data']
            labels[person_id]['overall_person_id'] = overall_person_id

        write_to_text(img_width, img_height, labels, label_fpath)
 
if __name__ == '__main__':
    root_dir = '/mnt/sda1/user/data/DAVID_893'
    ann_root = f'{root_dir}/jsons'
    train_label_dir = f'{root_dir}/labels_with_ids_both_vh/train'
    val_label_dir = f'{root_dir}/labels_with_ids_both_vh/val'
    train_file_list = f'{root_dir}/train.txt'
    val_file_list = f'{root_dir}/val.txt'
    
    gen_labels_david(train_file_list, train_label_dir, ann_root)
    gen_labels_david(val_file_list, val_label_dir, ann_root)