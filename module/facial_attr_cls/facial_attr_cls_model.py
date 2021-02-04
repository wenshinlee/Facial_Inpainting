import torch
import torch.nn as nn

import os
import cv2
import numpy as np
from itertools import chain
from collections import OrderedDict

import options
import module.networks as networks
from module.facial_attr_cls.resnet import resnet34


class FacialAttrCls:
    def __init__(self, is_train, facial_attr_info_dict, checkpoints_dir='./checkpoints'):
        super(FacialAttrCls, self).__init__()
        opt = options.TrainOption().parse()
        self.is_train = is_train
        self.checkpoints_dir = checkpoints_dir
        self.model_names = []
        self.optimizers = []
        self.schedulers = []
        self.image = None
        self.attr = None
        self.filename = None
        self.loss = None
        self.selected_attrs = list(chain.from_iterable(facial_attr_info_dict['each_facial_fea_attr']))

        self.model = resnet34(pretrained=False, facial_info=facial_attr_info_dict)
        self.model_names.append('model')

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device=self.device)

        if self.is_train:
            self.criterion = nn.BCEWithLogitsLoss()
            self.optimizer_model = torch.optim.Adam(self.model.parameters(),
                                                    lr=0.0001, betas=(0.5, 0.999))
            self.optimizers.append(self.optimizer_model)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

    def set_input(self, inputs):
        filename, image, attr = inputs
        self.image = image.to(self.device)
        self.attr = attr.to(self.device)
        self.filename = filename

    def forward(self):
        self.ax_prob, self.result, self.att = self.model(self.image)

    def backward(self):
        self.loss1 = self.criterion(self.ax_prob, self.attr)
        self.loss2 = self.criterion(self.result, self.attr)
        self.loss = self.loss1 + self.loss2
        self.loss.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_model.zero_grad()
        self.backward()
        self.optimizer_model.step()

    def get_current_errors(self):
        return OrderedDict([('loss1', self.loss1),
                            ('loss2', self.loss2),
                            ('loss', self.loss)])

    def get_current_visuals(self, which_epoch):
        self.vis_actmap(self.att, self.image, which_epoch)

    def vis_actmap(self, att, image, which_epoch):
        b, c, h, w = att.shape
        for i in range(b):
            for j in range(c):
                # org_image
                org_img = ((image[i].detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2.0 * 255).astype(np.uint8)
                org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
                # attention map
                facial_map = (att[i, j].detach().cpu().numpy() * 255).astype(np.uint8)
                heat_img = cv2.resize(facial_map, (org_img.shape[0], org_img.shape[1]))
                heat_img = cv2.applyColorMap(heat_img, cv2.COLORMAP_JET)

                img_add = cv2.addWeighted(org_img, 0.3, heat_img, 0.7, 0)

                filename = os.path.join(self.checkpoints_dir, 'pic', str(which_epoch) +
                                        str(self.filename[i]).split('.')[0] + '_' +
                                        self.selected_attrs[j] + '.jpg')
                cv2.imwrite(filename, img_add)

    def save_networks(self, which_epoch, checkpoints_dir):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_resnet34.pth' % which_epoch
                save_path = os.path.join(checkpoints_dir, save_filename)
                net = getattr(self, name)
                torch.save(net.state_dict(), save_path)

    def load_networks(self, which_epoch, checkpoints_dir):
        for name in self.model_names:
            if isinstance(name, str):
                # load pretrain model
                load_filename = '%s_resnet34.pth' % which_epoch
                load_path = os.path.join(checkpoints_dir, load_filename)
                self.model.load_state_dict(torch.load(load_path))
            print("load [%s] successful!" % name)

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def test(self, which_epoch, checkpoints_dir):
        self.load_networks(which_epoch=which_epoch, checkpoints_dir=checkpoints_dir)
        self.forward()
        self.get_current_visuals()
