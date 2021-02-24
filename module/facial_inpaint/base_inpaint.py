import os
from abc import ABC

import torch
import util.utils as utils


class BaseNetwork(torch.nn.Module, ABC):
    def __init__(self, opt):
        super(BaseNetwork, self).__init__()
        self.opt = opt
        self.is_train = opt.is_train
        if not self.is_train:
            utils.mkdirs(self.opt.results_fake_dir)
            utils.mkdirs(self.opt.results_gt_dir)
            utils.mkdirs(self.opt.results_input_dir)
            utils.mkdirs(self.opt.results_mask_dir)
        # list
        self.model_names = []
        self.optimizers = []
        self.schedulers = []
        # device
        self.device = torch.device('cuda:{}'.format(self.opt.gpu_ids[0])) if self.opt.gpu_ids else torch.device('cpu')

        # define_loss_names
        # generator
        self.loss_G = 0
        # base generator loss
        self.loss_L1 = 0
        self.Perceptual_loss = 0
        self.Style_Loss = 0
        # Global/Local(optional)/Base generator loss
        self.loss_G_Global = 0
        self.loss_G_Local = 0
        self.loss_G_GAN = 0
        # Classified loss(optional)
        self.loss_G_real_CLA = 0
        self.loss_G_fake_CLA = 0
        self.loss_G_CLA = 0

        # discriminator
        self.loss_D = 0
        # Global/Local(optional)/Base discriminator loss
        self.loss_D_Global = 0
        self.loss_D_Local = 0
        self.loss_D_GAN = 0
        # Classified loss(optional)
        self.loss_D_fake_CLA = 0
        self.loss_D_real_CLA = 0
        self.loss_D_CLA = 0

    def save_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_%s_%s.pth' % (self.opt.model, which_epoch, name)
                save_path = os.path.join(self.opt.checkpoints_dir, save_filename)
                net = getattr(self, name)
                optimize = getattr(self, 'optimizer_' + name)

                if len(self.opt.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save({'net': net.module.cpu().state_dict(), 'optimize': optimize.state_dict()}, save_path)
                    net.cuda(self.opt.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def load_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                # load pretrain model
                load_filename = '%s_%s_%s.pth' % (self.opt.model, which_epoch, name)
                load_path = os.path.join(self.opt.checkpoints_dir, load_filename)
                state_dict = torch.load(load_path, map_location=str(self.device))

                net = getattr(self, name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                net.load_state_dict(state_dict['net'])

                if self.is_train:
                    optimize = getattr(self, 'optimizer_' + name)
                    optimize.load_state_dict(state_dict['optimize'])

            print("load [%s] successful!" % name)

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        """Set requires_grad=False for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
