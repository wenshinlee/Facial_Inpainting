import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler

import functools
from module.facial_inpaint.architecture import Generator, NLayerDiscriminator, NLayerClassifierDiscriminator


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=None):
    if gpu_ids is None:
        gpu_ids = []
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number of parameters: %d' % num_params)


def define_G_DMFB(facial_fea_names, facial_fea_attr_names, facial_fea_attr_len, add_noise=False, spade_segmap=True,
                  latent_vector_size=512, skip_type='res', region_encoder=True, is_spectral_norm=True, gpu_ids=None,
                  norm_type='instance', init_type='normal', init_gain=0.02):
    if gpu_ids is None:
        gpu_ids = []
    norm_layer = get_norm_layer(norm_type=norm_type)
    net = Generator(facial_fea_names, facial_fea_attr_names, facial_fea_attr_len, add_noise, spade_segmap,
                    latent_vector_size, skip_type, region_encoder, is_spectral_norm, norm_layer=norm_layer)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D_DMFB(input_nc=3, gpu_ids=None, norm_type='batch', init_type='normal', init_gain=0.02):
    if gpu_ids is None:
        gpu_ids = []
    norm_layer = get_norm_layer(norm_type=norm_type)
    netD = NLayerDiscriminator(input_nc, ndf=64, n_layers=3, norm_layer=norm_layer)
    return init_net(netD, init_type, init_gain, gpu_ids)


def define_D_Classifier_DMFB(facial_fea_names, facial_fea_attr_len, gpu_ids=None, use_spectral_norm=False,
                             use_sigmoid=False, init_type='normal', init_gain=0.02):
    if gpu_ids is None:
        gpu_ids = []
    netD = NLayerClassifierDiscriminator(facial_fea_names, facial_fea_attr_len, input_nc=3,
                                         use_sigmoid=use_sigmoid, use_spectral_norm=use_spectral_norm)
    return init_net(netD, init_type, init_gain, gpu_ids)
