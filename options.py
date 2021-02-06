import os
import torch
from util import utils
import argparse


class BaseOption(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.is_train = None

    def initialize(self, parser):
        # base define
        parser.add_argument('--model', type=str, default='new_attr_model', help='name of the model type.')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                            help='models are saved here and options')
        # facial yaml
        parser.add_argument('--yaml_config_path', type=str,
                            default='./datasets/facial_feature.yaml',
                            help='yaml file')
        parser.add_argument('--dataset_shuffle', action='store_false', default=True,
                            help='Whether to disrupt the order of data sets,default is True')
        parser.add_argument('--dataset_shuffle_seed', type=int, default=123,
                            help='shuffle seed')
        # dataset
        parser.add_argument('--image_dir', type=str,
                            default='/home/datasets/inpaint/celeba/Celeba-HQ-inpaint/images',
                            help='dir to detail images (which are the groundtruth)')
        parser.add_argument('--mask_dir', type=str,
                            default='/home/datasets/inpaint/celeba/Celeba-HQ-inpaint/mask',
                            help='dir to mask (celeba-HQ-mask)')
        parser.add_argument('--p_irregular_miss', type=int, default=0.5,
                            help='max miss number')
        parser.add_argument('--max_num_miss', type=int, default=4,
                            help='max miss number')
        parser.add_argument('--image_size', type=int, default=256,
                            help='then crop and resize to this size')
        parser.add_argument('--dataset_name', type=str,
                            default='CelebA_Attr_CV2', help='which datasets, CelebA_HQ_Mask_CV2/CelebA_HQ_Mask_Skin')
        parser.add_argument('--dilate_iter', type=int, default=2,
                            help='dilate iter')
        parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
        parser.add_argument('--num_workers', type=int, default=4, help='numbers of the core of CPU')

        # generator
        parser.add_argument('--add_noise', action='store_false', default=True,
                            help='AttrMaskNorm input x add noise')
        parser.add_argument('--spade_segmap', action='store_false', default=True,
                            help='spade input segmap or mask')
        parser.add_argument('--skip_type', type=str, default='learned',
                            help='The original residual network method is forced to be used for jump connection,'
                                 'can be learned or original')
        parser.add_argument('--latent_vector_size', type=int, default=512,
                            help='style/attr_encoder latent vector size')
        parser.add_argument('--region_encoder', action='store_true', default=False,
                            help='region_encoder model and attr_encoder model')
        parser.add_argument('--is_spectral_norm', action='store_true', default=False,
                            help='apply Spectral Normalization in decoder')
        parser.add_argument('--gpu_ids', type=str, default='0',
                            help='gpu ids: e.g. 0, 1, 2 use -1 for CPU')
        parser.add_argument('--norm_type', type=str, default='instance',
                            help='instance/batch/none')
        # Discriminator
        parser.add_argument('--local_dis', action='store_true', default=False,
                            help='local discriminator')
        # optimizers
        parser.add_argument('--lr', type=float, default=0.0001,
                            help='initial learning rate for adam')
        parser.add_argument('--adam_beta', type=float, default=0.5,
                            help='momentum term of adam')
        parser.add_argument('--lr_policy', type=str, default='lambda',
                            help='learning rate policy:lambda|step|plateau|cosine')
        parser.add_argument('--lr_decay_iters', type=int, default=50,
                            help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--epoch_count', type=int, default=1,
                            help='the starting epoch count, we save the model by <epoch_count>, '
                                 '<epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--niter', type=int, default=20,
                            help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=100,
                            help='# of iter to linearly decay learning rate to zero')

        # model init
        parser.add_argument('--init_type', type=str, default='normal',
                            help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02,
                            help='scaling factor for normal, xavier and orthogonal.')
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = self.initialize(self.parser)
            opt = parser.parse_args()
            return opt
        else:
            raise NotImplemented('options init fail.')

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, 'options')
        if not os.path.exists(expr_dir):
            utils.mkdirs(expr_dir)

        if opt.is_train:
            file_name = os.path.join(expr_dir, opt.model + '_train_opt.txt')
        else:
            file_name = os.path.join(expr_dir, opt.model + '_test_opt.txt')

        with open(file_name, 'wt') as opt_file:
            opt_file.write('--------------Options--------------\n')
            for k, v in sorted(vars(opt).items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('----------------End----------------\n')

    def parse(self):
        opt = self.gather_options()
        opt.is_train = self.is_train

        # set gpu ids(convert str to list)
        str_ids = str(opt.gpu_ids).split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
            else:
                opt.gpu_ids.clear()
                opt.gpu_ids.append(-1)
                break
        if len(opt.gpu_ids):
            torch.cuda.set_device(opt.gpu_ids[0])

        # print options
        self.print_options(opt)

        return opt


class TrainOption(BaseOption):
    def __init__(self):
        super(TrainOption, self).__init__()

    def initialize(self, parser):
        super(TrainOption, self).initialize(parser)
        parser.add_argument('--continue_train', action='store_true', default=False,
                            help='continue training: load the latest model')
        parser.add_argument('--which_epoch', type=str, default='',
                            help='which epoch to load? set to latest to use latest cached model')
        # visualization
        parser.add_argument('--display_freq', type=int, default=150,
                            help='frequency of showing training results on tensorboard image')
        parser.add_argument('--print_freq', type=int, default=50,
                            help='frequency of showing training results on tensorboard scalar')
        # save model
        parser.add_argument('--save_epoch_freq', type=int, default=1,
                            help='frequency of saving checkpoints at the end of epochs')

        self.is_train = True
        return parser


class TestOption(BaseOption):
    def __init__(self):
        super(TestOption, self).__init__()

    def initialize(self, parser):
        super(TestOption, self).initialize(parser)
        parser.add_argument('--which_epoch', type=str, default='',
                            help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--results_fake_dir', type=str, default='./output/fake_img',
                            help='')
        parser.add_argument('--results_gt_dir', type=str, default='./output/gt_img',
                            help='')
        parser.add_argument('--results_input_dir', type=str, default='./output/input_img',
                            help='')
        self.is_train = False
        return parser
