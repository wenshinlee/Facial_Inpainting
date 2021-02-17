from abc import ABC

import torch
import torch.nn as nn
import random
from module.facial_inpaint.base_inpaint import BaseNetwork
from module import networks as networks
from module.loss.loss import RelativisticAverageLoss, PerceptualLoss, StyleLoss
# test
import numpy as np
import os
import cv2


class FacialInpaint(BaseNetwork, ABC):
    def __init__(self, opt, facial_fea_names, facial_fea_attr_names, facial_fea_attr_len):
        super(FacialInpaint, self).__init__(opt=opt)
        self.facial_fea_attr_len = facial_fea_attr_len
        self.region_encoder = opt.region_encoder

        self.G = networks.define_G_DMFB(facial_fea_names, facial_fea_attr_names, facial_fea_attr_len, opt.add_noise,
                                        opt.spade_segmap, opt.latent_vector_size, opt.skip_type, opt.region_encoder,
                                        opt.is_spectral_norm, opt.gpu_ids, opt.norm_type, opt.init_type, opt.init_gain)
        self.model_names.append('G')

        if self.is_train:
            if opt.local_dis:
                self.L = networks.define_D_DMFB(input_nc=3, gpu_ids=opt.gpu_ids, norm_type=opt.norm_type)
                self.model_names.append('L')
            if self.region_encoder:
                self.D = networks.define_D_DMFB(input_nc=3, gpu_ids=opt.gpu_ids, norm_type=opt.norm_type)
            else:
                self.D = networks.define_D_Classifier_DMFB(facial_fea_names, facial_fea_attr_len,
                                                           gpu_ids=opt.gpu_ids)
            self.model_names.append('D')

        if self.is_train:
            # define loss functions
            self.criterionGAN = RelativisticAverageLoss()
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.PerceptualLoss = PerceptualLoss()
            self.StyleLoss = StyleLoss()
            # optional field
            if not self.region_encoder:
                self.criterionClassifier = nn.BCEWithLogitsLoss()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.G.parameters(),
                                                lr=opt.lr, betas=(opt.adam_beta, 0.999))
            self.optimizer_D = torch.optim.Adam(self.D.parameters(),
                                                lr=opt.lr, betas=(opt.adam_beta, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            # optional field
            if opt.local_dis:
                self.optimizer_L = torch.optim.Adam(self.L.parameters(),
                                                    lr=opt.lr, betas=(opt.adam_beta, 0.999))
                self.optimizers.append(self.optimizer_L)

            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

            print('---------- Networks initialized -------------')
            networks.print_network(self.G)
            networks.print_network(self.D)
            # optional field
            if opt.local_dis:
                networks.print_network(self.L)
            print('-----------------------------------------------')

            if opt.continue_train:
                print('Loading pre-trained network for train!')
                self.load_networks(opt.which_epoch)

        if not self.is_train:
            print('Loading pre-trained network for test!')
            self.load_networks(opt.which_epoch)
            for model in self.model_names:
                getattr(self, model).eval()
                for para in getattr(self, model).parameters():
                    para.requires_grad = False

    def set_input(self, inputs):
        file_name, image_gt, mask, segmap, attr_matrix = inputs

        self.file_name = file_name
        self.image_gt = image_gt.to(self.device)
        self.segmap = segmap.to(self.device)
        self.attr_matrix = attr_matrix.to(self.device)

        # define local area which send to the local discriminator
        if self.opt.local_dis:
            self.local_gt = image_gt.to(self.device)
            self.crop_x = random.randint(0, 191)
            self.crop_y = random.randint(0, 191)
            self.local_gt = self.local_gt[:, :, self.crop_x:self.crop_x + 64, self.crop_y:self.crop_y + 64]

        # mask
        self.mask = mask.to(self.device)

        # mask 0 is hole
        self.inv_ex_mask = torch.add(torch.neg(self.mask.float()), 1).float()

        # Do not set the mask regions as 0
        self.input = image_gt.to(self.device)
        self.input.narrow(1, 0, 1).masked_fill_(self.mask.narrow(1, 0, 1).bool(), 2 * 123.0 / 255.0 - 1.0)
        self.input.narrow(1, 1, 1).masked_fill_(self.mask.narrow(1, 0, 1).bool(), 2 * 104.0 / 255.0 - 1.0)
        self.input.narrow(1, 2, 1).masked_fill_(self.mask.narrow(1, 0, 1).bool(), 2 * 117.0 / 255.0 - 1.0)

    def forward(self):
        self.fake_out = self.G(self.input, self.segmap, self.mask, self.attr_matrix)

    def backward_d(self):
        real = self.image_gt
        fake = self.fake_out

        # Global Discriminator
        pred_fake, cla_fake = self.D(fake.detach(), self.segmap)
        pred_real, cla_real = self.D(real, self.segmap)
        self.loss_D_Global = self.criterionGAN(pred_real, pred_fake, True)

        # optional field
        # Local Discriminator
        if self.opt.local_dis:
            real_local = self.local_gt
            fake_local = self.fake_out[:, :, self.crop_x:self.crop_x + 64, self.crop_y:self.crop_y + 64]
            pred_fake_l, _ = self.L(fake_local.detach())
            pred_real_l, _ = self.L(real_local)
            self.loss_D_Local = self.criterionGAN(pred_fake_l, pred_real_l, True)

        # finally Generator loss
        self.loss_D_GAN = self.loss_D_Global + self.loss_D_Local

        # optional field
        # Discriminator classified loss
        if not self.region_encoder:
            self.loss_D_fake_CLA = self.get_cla_loss(cla_fake)
            self.loss_D_real_CLA = self.get_cla_loss(cla_real)
            self.loss_D_CLA = self.loss_D_real_CLA + self.loss_D_fake_CLA

        self.loss_D = self.loss_D_GAN + self.loss_D_CLA
        self.loss_D.backward()

    def backward_g(self):
        real = self.image_gt
        fake = self.fake_out

        # First, Reconstruction loss, style loss, L1 loss
        self.loss_L1 = self.criterionL1(fake, real)
        self.Perceptual_loss = self.PerceptualLoss(fake, real)
        self.Style_Loss = self.StyleLoss(fake, real)

        # Second, The generator should fake the discriminator
        # Global discriminator
        pred_real, cla_real = self.D(real, self.segmap)
        pred_fake, cla_fake = self.D(fake, self.segmap)
        self.loss_G_Global = self.criterionGAN(pred_real, pred_fake, False)

        # optional field
        # Local discriminator
        if self.opt.local_dis:
            real_local = self.local_gt
            fake_local = self.fake_out[:, :, self.crop_x:self.crop_x + 64, self.crop_y:self.crop_y + 64]
            pred_real_l, _ = self.L(real_local)
            pred_fake_l, _ = self.L(fake_local)
            self.loss_G_Local = self.criterionGAN(pred_real_l, pred_fake_l, False)

        self.loss_G_GAN = self.loss_G_Global + self.loss_G_Local

        # Third, Generator classified loss
        if not self.region_encoder:
            self.loss_G_fake_CLA = self.get_cla_loss(cla_fake)
            self.loss_G_real_CLA = self.get_cla_loss(cla_real)
            self.loss_G_CLA = self.loss_G_real_CLA + self.loss_G_fake_CLA

        # finally Generator loss
        self.loss_G = self.loss_L1 * 1 + self.Perceptual_loss * 0.2 + self.Style_Loss * 250 \
                      + self.loss_G_GAN * 0.2 + self.loss_G_CLA
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # Optimize the D and L first
        self.set_requires_grad(self.G, False)
        self.set_requires_grad(self.D, True)
        self.optimizer_D.zero_grad()
        if self.opt.local_dis:
            self.set_requires_grad(self.L, True)
            self.optimizer_L.zero_grad()

        self.backward_d()
        self.optimizer_D.step()
        if self.opt.local_dis:
            self.optimizer_L.step()

        # Optimize G
        self.set_requires_grad(self.G, True)
        self.set_requires_grad(self.D, False)
        if self.opt.local_dis:
            self.set_requires_grad(self.L, False)
        self.optimizer_G.zero_grad()
        self.backward_g()
        self.optimizer_G.step()

    def get_current_errors(self):
        # show the current loss
        loss_dict = {
            # discriminator
            'loss_D_Global': self.loss_D_Global.data,
            'loss_D_GAN': self.loss_D_GAN.data,
            'loss_D': self.loss_D.data,
            # Generator
            'loss_G_Global': self.loss_G_Global.data,
            'loss_G_GAN': self.loss_G_GAN,
            'loss_L1': self.loss_L1.data,
            'Perceptual_loss': self.Perceptual_loss.data,
            'Style_Loss': self.Style_Loss.data,
            'loss_G': self.loss_G.data
        }
        if self.opt.local_dis:
            loss_dict.update({"loss_D_Local": self.loss_D_Local})
            loss_dict.update({"loss_G_Local": self.loss_G_Local})
        if not self.region_encoder:
            loss_dict.update({"loss_G_fake_CLA": self.loss_G_fake_CLA.data})
            loss_dict.update({"loss_G_real_CLA": self.loss_G_real_CLA.data})
            loss_dict.update({"loss_D_fake_CLA": self.loss_D_fake_CLA.data})
            loss_dict.update({"loss_D_real_CLA": self.loss_D_real_CLA.data})

        return loss_dict

    def get_current_visuals(self):
        input_image = (self.input.data.cpu() + 1) / 2.0
        fake_image = (self.fake_out.data.cpu() + 1) / 2.0
        real_gt = (self.image_gt.data.cpu() + 1) / 2.0
        return input_image, fake_image, real_gt

    def get_cla_loss(self, net_out):
        cla_loss = 0
        for idx, facial_fea_cls_pred in enumerate(net_out):
            cla_loss = cla_loss + self.criterionClassifier(facial_fea_cls_pred,
                                                           self.attr_matrix[:, idx, 0:self.facial_fea_attr_len[idx]])
        return cla_loss

    def test(self):
        self.forward()
        input_image, fake_image, real_gt = self.get_current_visuals()
        # chw -> hwc
        gt = real_gt.numpy().transpose((0, 2, 3, 1)) * 255
        output = fake_image.numpy().transpose((0, 2, 3, 1)) * 255
        inputs = input_image.numpy().transpose((0, 2, 3, 1)) * 255
        for idx in range(output.shape[0]):
            # rgb -> bgr
            save_gt = gt[idx][..., ::-1].astype(np.uint8)
            save_output = output[idx][..., ::-1].astype(np.uint8)
            save_inputs = inputs[idx][..., ::-1].astype(np.uint8)
            # save path
            fake_save_path = os.path.join(self.opt.results_fake_dir, self.file_name[idx])
            gt_save_path = os.path.join(self.opt.results_gt_dir, self.file_name[idx])
            inputs_save_path = os.path.join(self.opt.results_input_dir, self.file_name[idx])
            # save
            cv2.imwrite(fake_save_path, save_output)
            cv2.imwrite(gt_save_path, save_gt)
            cv2.imwrite(inputs_save_path, save_inputs)


