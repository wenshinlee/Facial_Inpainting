import time
import tqdm
import torch
import torchvision

import options
from datasets.dataset import get_dataloader
from module.facial_inpaint.facial_inpaint import FacialInpaint
from util.visuals import Visuals
from datasets.facial_yaml import FacialYaml


if __name__ == '__main__':
    opt = options.TrainOption().parse()
    # facial yaml
    facial_yaml = FacialYaml(opt)
    facial_fea_names = facial_yaml.get_facial_fea_names()
    facial_fea_attr_names = facial_yaml.get_facial_fea_attr_names()
    facial_fea_attr_len = facial_yaml.get_facial_fea_attr_len()
    facial_attr_dataset = facial_yaml.get_facial_attr_dataset()

    data_loader = get_dataloader(opt.image_dir, opt.segmap_mask_dir, opt.pconv_mask_dir, facial_fea_names,
                                 facial_fea_attr_names, facial_fea_attr_len, facial_attr_dataset,
                                 opt.p_generate_miss, opt.max_num_miss, opt.image_size, opt.dataset_name,
                                 opt.dilate_iter, opt.is_train, opt.batch_size, opt.num_workers)

    model = FacialInpaint(opt, facial_fea_names, facial_fea_attr_names, facial_fea_attr_len)
    visuals = Visuals(opt.checkpoints_dir, opt.model)

    total_steps = 0
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        for n_iter, data in tqdm.tqdm(enumerate(data_loader, 0), total=len(data_loader),
                                      desc="epoch-->%d" % epoch, ncols=80, leave=False):
            iter_start_time = time.time()

            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            # display the training processing
            if epoch_iter % opt.display_freq == 0:
                input_image, fake_image, real_gt = model.get_current_visuals()
                image_out = torch.cat([input_image, fake_image, real_gt], 0)
                grid = torchvision.utils.make_grid(image_out, nrow=input_image.shape[0])
                visuals.add_image('Epoch_(%d)_(%d)' % (epoch, epoch_iter + 1), grid, total_steps + 1)

            if epoch_iter % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = round((time.time() - iter_start_time) / opt.batch_size, 4)
                # discriminator
                visuals.add_scalar('discriminator/loss_D_Global', errors['loss_D_Global'], total_steps + 1)
                visuals.add_scalar('discriminator/loss_D_GAN', errors['loss_D_GAN'], total_steps + 1)
                visuals.add_scalar('discriminator/loss_D', errors['loss_D'], total_steps + 1)

                # generator
                visuals.add_scalar('generator/loss_G_Global', errors['loss_G_Global'], total_steps + 1)
                visuals.add_scalar('generator/loss_G_GAN', errors['loss_G_GAN'], total_steps + 1)
                visuals.add_scalar('generator/loss_L1', errors['loss_L1'], total_steps + 1)
                visuals.add_scalar('generator/Perceptual_loss', errors['Perceptual_loss'], total_steps + 1)
                visuals.add_scalar('generator/Style_Loss', errors['Style_Loss'], total_steps + 1)
                visuals.add_scalar('generator/loss_G', errors['loss_G'], total_steps + 1)
                if opt.local_dis:
                    visuals.add_scalar('discriminator/loss_D_Local', errors['loss_D_Local'], total_steps + 1)
                    visuals.add_scalar('generator/loss_G_Local', errors['loss_G_Local'], total_steps + 1)
                if not opt.region_encoder:
                    visuals.add_scalar('generator/loss_G_fake_CLA', errors['loss_G_fake_CLA'], total_steps + 1)
                    visuals.add_scalar('generator/loss_G_real_CLA', errors['loss_G_real_CLA'], total_steps + 1)
                    visuals.add_scalar('discriminator/loss_D_fake_CLA', errors['loss_D_fake_CLA'], total_steps + 1)
                    visuals.add_scalar('discriminator/loss_D_real_CLA', errors['loss_D_real_CLA'], total_steps + 1)

                print('iteration time: %f' % t)

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks(epoch)
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

    print("Finish!")
