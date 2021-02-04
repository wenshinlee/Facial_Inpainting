import tqdm

from datasets.dataset import get_dataloader
from util.visuals import Visuals
from module.facial_attr_cls.facial_attr_cls_model import FacialAttrCls
from datasets.facial_yaml import FacialYaml


def train(facial_attr_info_dict):
    image_dir = '/home/datasets/inpaint/celeba/Celeba-HQ-inpaint/images'
    dataset_name = 'CelebA_Face_Cls'
    batch_size = 64
    data_loader = get_dataloader(image_dir=image_dir, mask_dir=None,
                                 facial_attr_info_dict=facial_attr_info_dict,
                                 dataset_name=dataset_name, batch_size=batch_size)
    checkpoints_dir = './checkpoints'
    model = FacialAttrCls(is_train=True, facial_attr_info_dict=facial_attr_info_dict, checkpoints_dir=checkpoints_dir)
    visuals = Visuals(checkpoints_dir, 'CelebA_Face_Cls')
    model.load_networks(120, checkpoints_dir)

    total_steps = 0
    print_freq = 640
    for epoch in range(0, 120):
        epoch_iter = 0
        for n_iter, data in tqdm.tqdm(enumerate(data_loader, 0), total=len(data_loader),
                                      desc="epoch-->%d" % epoch, ncols=80, leave=False):

            total_steps += batch_size
            epoch_iter += batch_size
            model.set_input(data)
            model.optimize_parameters()

            if epoch_iter % print_freq == 0:
                errors = model.get_current_errors()
                visuals.add_scalar('loss', errors['loss'], total_steps + 1)
                visuals.add_scalar('loss1', errors['loss1'], total_steps + 1)
                visuals.add_scalar('loss2', errors['loss2'], total_steps + 1)
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, epoch_iter, errors['loss']))

        if (epoch + 1) % 10 == 0:
            model.get_current_visuals(epoch+1)
            model.save_networks(epoch + 1, checkpoints_dir)
        model.update_learning_rate()
    print('Finished Training')


def test(facial_attr_info_dict):
    image_dir = '/home/datasets/inpaint/celeba/Celeba-HQ-inpaint/images'
    dataset_name = 'CelebA_Face_Cls'
    batch_size = 2
    checkpoints_dir = 'checkpoints'
    data_loader = get_dataloader(image_dir=image_dir, mask_dir=None,
                                 facial_attr_info_dict=facial_attr_info_dict,
                                 dataset_name=dataset_name, batch_size=batch_size, is_train=True)
    model = FacialAttrCls(is_train=False, facial_attr_info_dict=facial_attr_info_dict, checkpoints_dir=checkpoints_dir)
    inputs = next(iter(data_loader))
    model.set_input(inputs)
    model.test(which_epoch=30, checkpoints_dir=checkpoints_dir)


if __name__ == '__main__':
    facial_fea_yaml = './datasets/facial_feature.yaml'
    facial_yaml = FacialYaml(facial_fea_yaml)
    facial_attr_info = facial_yaml.return_facial_attr_info_dict()

    # test(facial_attr_info)
    train(facial_attr_info)
