from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as models
import torch
import torch.nn as nn
import os

from .networks.dlav0 import get_pose_net as get_dlav0
from .networks.pose_dla_dcn import get_pose_net as get_dla_dcn
from .networks.resnet_dcn import get_pose_net as get_pose_net_dcn
from .networks.resnet_fpn_dcn import get_pose_net as get_pose_net_fpn_dcn
from .networks.pose_hrnet import get_pose_net as get_pose_net_hrnet
from .networks.pose_dla_conv import get_pose_net as get_dla_conv
import sys
sys.path.append('..')
from trains.mot import MotLoss

_model_factory = {
  'dlav0': get_dlav0, # default DLAup
  'dla': get_dla_dcn,
  'dlaconv': get_dla_conv,
  'resdcn': get_pose_net_dcn,
  'resfpndcn': get_pose_net_fpn_dcn,
  'hrnet': get_pose_net_hrnet
}

def create_model(arch, heads, head_conv):
  num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
  arch = arch[:arch.find('_')] if '_' in arch else arch
  get_model = _model_factory[arch]
  model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)
  return model

def load_model(trainer, model_path, resume=False,
               lr=None, lr_step=None):
  start_epoch = 0
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  print('trying to load in {}, epoch {}'.format(model_path, checkpoint['epoch']))
  state_dict_ = checkpoint['state_dict']
  state_dict = {}

  model = trainer.model
  if 'id_clf' in checkpoint.keys():
    id_state_dict = checkpoint['id_clf']
    print(f'id_state_dict type : {type(id_state_dict)}')
    if isinstance(id_state_dict, MotLoss):
      id_state_dict = id_state_dict.classifier.state_dict()
      trainer.loss.classifier.load_state_dict(id_state_dict,  strict=False)
    else: print("we found 'id_clf' but coudn't load state dict for some reasons")
  else : print("coudn't find ID_CLF in checkpoint. ID_CLF will start from scratch" )

  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]
  model_state_dict = model.state_dict()

  # check loaded parameters and created model parameters
  msg = 'If you see this, your model does not fully load the ' + \
        'pre-trained weight. Please make sure ' + \
        'you have correctly specified --arch xxx ' + \
        'or set the correct --num_classes for your own dataset.'
  for k in state_dict:
    if k in model_state_dict:
      if state_dict[k].shape != model_state_dict[k].shape:
        print('Skip loading parameter {}, required shape{}, '\
              'loaded shape{}. {}'.format(
          k, model_state_dict[k].shape, state_dict[k].shape, msg))
        state_dict[k] = model_state_dict[k]
    else:
      print('Drop parameter {}.'.format(k) + msg)
  for k in model_state_dict:
    if not (k in state_dict):
      print('No param {}.'.format(k) + msg)
      state_dict[k] = model_state_dict[k]
  model.load_state_dict(state_dict, strict=False)
  # model.load_state_dict(state_dict_, strict=False)

  print('load process end')

  # resume optimizer parameters
  if resume:
    if 'optimizer' in checkpoint:
      trainer.optimizer.load_state_dict(checkpoint['optimizer'])
      start_epoch = checkpoint['epoch']
      start_lr = lr
      for step in lr_step:
        if start_epoch >= step:
          start_lr *= 0.1
      for param_group in trainer.optimizer.param_groups:
        param_group['lr'] = start_lr
      print(f'Let`s continue training from previous epoch : {start_epoch + 1}')
      print('Resumed optimizer with start lr', start_lr)
    else:
      print('No optimizer parameters in checkpoint.')

    return trainer, start_epoch
  else:
    return trainer

def save_model(path, epoch, trainer, optimizer=None):
  model = trainer.model
  id_clf = trainer.loss.classifier

  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
    id_clf = id_clf.module.state_dict()
  else:
    state_dict = model.state_dict()
    id_clf = id_clf.state_dict()
  data = {'epoch': epoch,
          'state_dict': state_dict,
          'id_clf' : id_clf}
  if not (optimizer is None):
    data['optimizer'] = optimizer.state_dict()
  torch.save(data, path)

