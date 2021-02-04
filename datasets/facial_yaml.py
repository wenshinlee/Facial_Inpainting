import os
import yaml
import numpy as np

from collections import OrderedDict


class FacialYaml:
    def __init__(self, opt):
        self.yaml_path = opt.yaml_config_path

        # shuffle
        self.dataset_shuffle = opt.dataset_shuffle
        self.dataset_shuffle_seed = opt.dataset_shuffle_seed
        # yaml
        self.yaml_cont_dict = self.load_yaml()
        self.support_facial_fea = self.get_support_facial_fea()
        self.selected_facial_fea_names = self.get_selected_facial_fea_names()
        self.selected_facial_fea_attr_names, self.selected_facial_fea_attr_len = self.get_selected_attrs()
        self.selected_facial_attrs_dataset = self.get_selected_attrs_dataset()

    def load_yaml(self):
        f = open(self.yaml_path, 'r', encoding='utf-8')
        yaml_cont_dict = yaml.load(f, Loader=yaml.FullLoader)
        return yaml_cont_dict

    def get_support_facial_fea(self, facial_fea_type='facial_fea_type'):
        if self.key_in_yaml(facial_fea_type):
            return self.yaml_cont_dict[facial_fea_type]
        else:
            raise ValueError('{} must in yaml!'.format(facial_fea_type))

    def key_in_yaml(self, key: str):
        return True if key in self.yaml_cont_dict else False

    def get_selected_facial_fea_names(self):
        selected_facial_fea_names = []
        for key, value in self.support_facial_fea.items():
            if value:
                selected_facial_fea_names.append(key)
        return selected_facial_fea_names

    def get_selected_attrs(self):
        selected_facial_fea_attr_names = []
        selected_facial_fea_attr_len = []
        for fea in self.selected_facial_fea_names:
            selected_attr_list = []
            attrs_len_counter = 0
            fea_dict = self.yaml_cont_dict.get(fea)
            for key, value in fea_dict.items():
                if value:
                    selected_attr_list.append(key)
                    attrs_len_counter += 1
            selected_facial_fea_attr_names.append(selected_attr_list)
            selected_facial_fea_attr_len.append(attrs_len_counter)
        return selected_facial_fea_attr_names, selected_facial_fea_attr_len

    def get_selected_attrs_dataset(self, attr_path_key='attr_path'):

        if self.key_in_yaml(attr_path_key):
            yaml_attr_path = self.yaml_cont_dict.get(attr_path_key)
            attr_txt_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], yaml_attr_path)
            lines = [line.rstrip() for line in open(attr_txt_path, 'r')]
            all_attr_names = lines[1].split()
            attr2idx = {}
            idx2attr = {}
            dataset = []

            for _, attr_name in enumerate(all_attr_names):
                attr2idx[attr_name] = _
                idx2attr[_] = attr_name

            lines = lines[2:]
            if self.dataset_shuffle:
                np.random.seed(self.dataset_shuffle_seed)
                np.random.shuffle(lines)

            for _, line in enumerate(lines):
                split = line.split()
                filename = split[0]
                values = split[1:]

                label = np.zeros((len(self.selected_facial_fea_names), max(self.selected_facial_fea_attr_len)),
                                 dtype=bool)
                for facial_fea_idx, attrs_names_list in enumerate(self.selected_facial_fea_attr_names):
                    for attr_idx, attr_names in enumerate(attrs_names_list):
                        idx = attr2idx[attr_names]
                        label[facial_fea_idx, attr_idx] = (values[idx] == '1')
                dataset.append([filename, label])

            print('Complete the preprocess of CelebA-HQ-MASK datasets...')

            return dataset
        else:
            raise ValueError('{} miss cant find attr file!'.format(attr_path_key))

    def get_facial_fea_names(self):
        return self.selected_facial_fea_names

    def get_facial_fea_attr_names(self):
        return self.selected_facial_fea_attr_names

    def get_facial_fea_attr_len(self):
        return self.selected_facial_fea_attr_len

    def get_facial_attr_dataset(self):
        return self.selected_facial_attrs_dataset

    def get_facial_attr_info_dict(self):
        return {'facial_fea_names': self.selected_facial_fea_names,
                'facial_fea_attr_names': self.selected_facial_fea_attr_names,
                'facial_fea_attr_len': self.selected_facial_fea_attr_len,
                'facial_attr_dataset': self.selected_facial_attrs_dataset
                }
