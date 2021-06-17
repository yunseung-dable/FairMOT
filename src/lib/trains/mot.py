from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models.losses import FocalLoss, TripletLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import mot_decode
from models.utils import _sigmoid, _tranpose_and_gather_feat
from utils.post_process import ctdet_post_process
from .base_trainer import BaseTrainer


class MotLoss(torch.nn.Module):
    def __init__(self, opt):
        super(MotLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt
        self.emb_dim = opt.reid_dim
        self.nID = opt.nID
        self.classifier = nn.Linear(self.emb_dim, self.nID)
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss, id_loss = 0, 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output['head_hm'] = _sigmoid(output['head_hm'])
                output['full_hm'] = _sigmoid(output['full_hm'])

            hm_loss += self.crit(output['head_hm'], batch['head_hm']) / opt.num_stacks
            hm_loss += self.crit(output['full_hm'], batch['full_hm']) / opt.num_stacks
            if opt.wh_weight > 0:
                wh_loss += self.crit_reg(
                    output['head_wh'], batch['head_reg_mask'], batch['head_ind'], batch['head_wh']) / opt.num_stacks
                wh_loss += self.crit_reg(
                    output['full_wh'], batch['full_reg_mask'], batch['full_ind'], batch['full_wh']) / opt.num_stacks
            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['head_reg'], batch['head_reg_mask'],batch['head_ind'], batch['head_reg']) / opt.num_stacks
                off_loss += self.crit_reg(output['full_reg'], batch['full_reg_mask'],batch['full_ind'], batch['full_reg']) / opt.num_stacks

            if opt.id_weight > 0: ####################### exclude head id_loss ######################

                # id_full(output)=이미지내에 오브젝트별 센터의 인덱스의 dimension만 가져옴 (bs, max_obj, dim)
                id_full = _tranpose_and_gather_feat(output['id'], batch['full_ind'])
                id_full = id_full[batch['full_reg_mask'] > 0].contiguous() # no obj filtering
                id_full = self.emb_scale * F.normalize(id_full)
                id_target = batch['ids'][batch['full_reg_mask'] > 0]

                id_output = self.classifier(id_full).contiguous()
                id_loss += self.IDLoss(id_output, id_target)

        det_loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss

        loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * id_loss + (self.s_det + self.s_id)
        # loss *= 0.5

        loss = det_loss + 0.1 * id_loss

        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss, 'id_loss': id_loss}
        return loss, loss_stats


class MotTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(MotTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'id_loss']
        loss = MotLoss(opt)
        return loss_states, loss

    def save_result(self, output, batch, results): ############################ head dets need to be fixed ############################
        # head_reg = output['head_reg'] if self.opt.reg_offset else None
        full_reg = output['full_reg'] if self.opt.reg_offset else None
        # head_dets = mot_decode(
        #     output['head_hm'], output['head_wh'], reg=head_reg,
        #     cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        full_dets = mot_decode(
            output['full_hm'], output['full_wh'], reg=full_reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)

        # head_dets = head_dets.cpu().detach().numpy().reshape(1, -1, head_dets.shape[2])
        full_dets = full_dets.cpu().detach().numpy().reshape(1, -1, full_dets.shape[2])

        # head_dets_out = ctdet_post_process(
        #     head_dets.copy(), batch['meta']['c'].cpu().numpy(),
        #     batch['meta']['s'].cpu().detach().numpy(),
        #     output['head_hm'].shape[2], output['head_hm'].shape[3], output['head_hm'].shape[1])
        # results[batch['meta']['img_id'].cpu().numpy()[0]] = head_dets_out[0]

        full_dets_out = ctdet_post_process(
            full_dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().detach().numpy(),
            output['full_hm'].shape[2], output['full_hm'].shape[3], output['full_hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = full_dets_out[0]
