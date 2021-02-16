import os
import cv2
import tqdm
import skimage.measure
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--path1', type=str, help='Path to the generated images')
parser.add_argument('--path2', type=str, help='Path to the groundtrue images')


def structural_similarity(image_true, image_test):
    return skimage.metrics.structural_similarity(image_true, image_test, multichannel=True)


def peak_signal_noise_ratio(image_true, image_test, data_range=None):
    return skimage.metrics.peak_signal_noise_ratio(image_true, image_test, data_range=data_range)


def mean_squared_error(image_true, image_test):
    return skimage.metrics.mean_squared_error(image_true, image_test)


def calculate_ssmi_given_paths(paths):
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)
    image_true = sorted(os.listdir(paths[0]))
    image_test = sorted(os.listdir(paths[1]))
    n_image_true = len(image_true)
    n_image_test = len(image_test)
    if n_image_test == n_image_true:
        p_value = 0.0
        s_value = 0.0
        m_value = 0.0
        for i in tqdm.tqdm(range(n_image_true)):
            image_true_ndarray = cv2.imread(os.path.join(paths[0], image_true[i]), cv2.IMREAD_COLOR)
            image_test_ndarray = cv2.imread(os.path.join(paths[1], image_test[i]), cv2.IMREAD_COLOR)
            s_value += structural_similarity(image_true_ndarray, image_test_ndarray)
            p_value += peak_signal_noise_ratio(image_true_ndarray, image_test_ndarray)
            m_value += mean_squared_error(image_true_ndarray, image_test_ndarray)
        return p_value / n_image_true, s_value / n_image_true, m_value/n_image_true
    else:
        raise ValueError('paths[0] and path[1] should have same number image!')


if __name__ == '__main__':
    """image name should same"""
    args = parser.parse_args()
    psnr, ssmi, mse = calculate_ssmi_given_paths([args.path1, args.path2])
    print("PSNR", psnr)
    print("SSMI", ssmi)
    print("MSE", mse)
