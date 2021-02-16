import tqdm

import options
from datasets.dataset import get_dataloader
from module.facial_inpaint.facial_inpaint import FacialInpaint
from datasets.facial_yaml import FacialYaml

if __name__ == '__main__':
    opt = options.TestOption().parse()
    # facial yaml
    facial_yaml = FacialYaml(opt)
    facial_fea_names = facial_yaml.get_facial_fea_names()
    facial_fea_attr_names = facial_yaml.get_facial_fea_attr_names()
    facial_fea_attr_len = facial_yaml.get_facial_fea_attr_len()
    facial_attr_dataset = facial_yaml.get_facial_attr_dataset()

    data_loader = get_dataloader(opt.image_dir, opt.test_mask_dir, facial_fea_names, facial_fea_attr_names,
                                 facial_fea_attr_len, facial_attr_dataset, opt.p_irregular_miss, opt.max_num_miss,
                                 opt.image_size, opt.test_dataset_name, opt.dilate_iter, opt.is_train,
                                 opt.batch_size, opt.num_workers)
    model = FacialInpaint(opt, facial_fea_names, facial_fea_attr_names, facial_fea_attr_len)

    for _, data in tqdm.tqdm(enumerate(data_loader, 0), total=len(data_loader),
                             desc="test ..ing", ncols=80, leave=False):
        model.set_input(data)
        model.test()

    print("finish!")
