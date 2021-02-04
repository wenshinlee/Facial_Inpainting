from abc import ABC

import torch
import torch.nn as nn
import torchvision.models as models


class VGG16(torch.nn.Module, ABC):
    def __init__(self):
        super(VGG16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.max3 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])
        for x in range(16, 17):
            self.max3.add_module(str(x), features[x])

        for x in range(17, 19):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(19, 21):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(23, 26):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(26, 28):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(28, 30):
            self.relu5_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        max_3 = self.max3(relu3_3)

        relu4_1 = self.relu4_1(max_3)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)

        relu5_1 = self.relu5_1(relu4_3)
        relu5_2 = self.relu5_1(relu5_1)
        relu5_3 = self.relu5_1(relu5_2)
        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'max_3': max_3,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
        }
        return out


class VGG19(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])
        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])
        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])
        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])
        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])
        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])
        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])
        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])
        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])
        print(self.relu1_1)
        print(self.relu1_2)
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        print(x.shape)
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        out = {
            'relu1_1': relu1_1,
            'relu2_1': relu2_1,
            'relu3_1': relu3_1,
            'relu4_1': relu4_1,
            'relu5_1': relu5_1
        }
        return out


class StyleLoss(nn.Module, ABC):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG16().cuda())
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_3']), self.compute_gram(y_vgg['relu3_3']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_3']), self.compute_gram(y_vgg['relu4_3']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return style_loss


class TVLoss(torch.nn.Module):
    """
    TV loss
    """

    def __init__(self, weight=1):
        super(TVLoss, self).__init__()
        self.weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class L1ReconLoss(torch.nn.Module):
    """
    L1 Reconstruction loss for two imgae
    """
    def __init__(self, weight=1):
        super(L1ReconLoss, self).__init__()
        self.weight = weight

    def forward(self, imgs, recon_imgs, masks=None):
        if masks is None:
            return self.weight * torch.mean(torch.abs(imgs - recon_imgs))
        else:
            return self.weight * torch.mean(
                torch.abs(imgs - recon_imgs) / masks.view(masks.size(0), -1).mean(1).view(-1, 1, 1, 1))


class PerceptualLoss(nn.Module, ABC):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=None):
        super(PerceptualLoss, self).__init__()
        if weights is None:
            weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.add_module('vgg', VGG16().cuda())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

        return content_loss


class RelativisticAverageLoss(nn.Module, ABC):
    def __init__(self):
        super(RelativisticAverageLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def __call__(self, real, fake, is_disc):
        avg_real = torch.mean(real)
        avg_fake = torch.mean(fake)
        ones = torch.ones_like(fake)
        zeros = torch.zeros_like(fake)
        if is_disc:
            return self.loss((real - avg_fake), ones) + self.loss((fake - avg_real), zeros)
        else:
            return self.loss((real - avg_fake), zeros) + self.loss((fake - avg_real), ones)


class SelfGuidedRegressionLoss(nn.Module, ABC):
    def __init__(self, weights=None):
        super().__init__()
        if weights is None:
            weights = [1.0, 1.0]
        self.weight = weights
        self.add_module('vgg19', VGG19().cuda())
        self.criterion = torch.nn.L1Loss()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def __call__(self, real, fake):
        # vgg out
        real_vgg, fake_vgg = self.vgg19(real), self.vgg19(fake)
        real_relu1_1 = real_vgg['relu1_1']
        fake_relu1_1 = fake_vgg['relu1_1']

        real_relu2_1 = real_vgg['relu2_1']
        fake_relu2_1 = fake_vgg['relu2_1']

        # guidance
        M_error = torch.sum((real - fake) ** 2, dim=1) / 3
        M_guidance1_1 = ((M_error - torch.min(M_error)) / (torch.max(M_error) - torch.min(M_error))).unsqueeze(1)
        M_guidance2_1 = self.avg_pool(M_guidance1_1)

        sgr_loss = 0.0
        sgr_loss += self.weight[0] * (1e3 / real_relu1_1.size()[1]) * self.criterion(M_guidance1_1 * real_relu1_1,
                                                                                     M_guidance1_1 * fake_relu1_1)

        sgr_loss += self.weight[1] * (1e3 / real_relu2_1.size()[1]) * self.criterion(M_guidance2_1 * real_relu2_1,
                                                                                     M_guidance2_1 * fake_relu2_1)
        return sgr_loss


class PerceptualLoss_VGG19(nn.Module, ABC):
    def __init__(self, weights=None):
        super().__init__()
        if weights is None:
            weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.weight = weights
        self.add_module('vgg19', VGG19().cuda())
        self.criterion = torch.nn.L1Loss()

    def __call__(self, real, fake):
        # vgg out
        real_vgg, fake_vgg = self.vgg19(real), self.vgg19(fake)

        loss = 0.0
        loss += self.weights[0] * self.criterion(real_vgg['relu1_1'], fake_vgg['relu1_1']) \
                * (1e3 / real_vgg['relu1_1'].size()[1])
        loss += self.weights[1] * self.criterion(real_vgg['relu2_1'], fake_vgg['relu2_1']) \
                * (1e3 / real_vgg['relu2_1'].size()[1])
        loss += self.weights[2] * self.criterion(real_vgg['relu3_1'], fake_vgg['relu3_1']) \
                * (1e3 / real_vgg['relu3_1'].size()[1])
        loss += self.weights[3] * self.criterion(real_vgg['relu4_1'], fake_vgg['relu4_1']) \
                * (1e3 / real_vgg['relu4_1'].size()[1])
        loss += self.weights[4] * self.criterion(real_vgg['relu5_1'], fake_vgg['relu5_1']) \
                * (1e3 / real_vgg['relu5_1'].size()[1])

        return loss


class RelativisticLoss(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def __call__(self, real, fake, is_disc):
        ones = torch.ones_like(fake)
        if is_disc:
            return self.loss((real - fake), ones)
        else:
            return self.loss((fake - real), ones)
