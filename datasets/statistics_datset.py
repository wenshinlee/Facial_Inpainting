from datasets.facial_yaml import FacialYaml
import numpy as np

facial_yaml = FacialYaml('facial_feature.yaml')
facial_dict = facial_yaml.return_facial_attr_info_dict()

selected_facial_fea = facial_dict['facial_fea']
selected_facial_fea_len = facial_dict['facial_fea_len']
selected_attrs = facial_dict['facial_fea_attr']
selected_attrs_dataset = facial_dict['attr_dataset']

list_arr_0 = []
list_arr_1 = []

for i in selected_attrs_dataset:
    if sum(np.array(i[1][0]).astype(int)) == 0:
        print("precessing {}".format(i[0]))
        print(np.array(i[1][0]).astype(int))
        list_arr_0.append(i[0])


print("------------")
print(list_arr_0)
print(len(list_arr_0))
# print(list_arr_0)
# print(len(list_arr_1))
# print(list_arr_1)
