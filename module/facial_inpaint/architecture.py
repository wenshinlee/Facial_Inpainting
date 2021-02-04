from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

import functools


class Generator(nn.Module, ABC):
    def __init__(self, facial_fea_names, facial_fea_attr_names, facial_fea_attr_len, add_noise=True,
                 latent_vector_size=512, region_encoder=True, is_spectral_norm=True, norm_layer=nn.InstanceNorm2d):
        super(Generator, self).__init__()
        self.region_encoder = region_encoder
        self.region_attr_encoder = RegionAttrEncoder(facial_fea_names, facial_fea_attr_names, facial_fea_attr_len,
                                                     region_encoder=region_encoder, norm_layer=norm_layer)
        self.decoder = Decoder(facial_fea_names, in_nc=512, out_nc=3, add_noise=add_noise,
                               latent_vector_size=latent_vector_size, region_normalized=region_encoder,
                               is_spectral_norm=is_spectral_norm, norm_layer=norm_layer)

    def forward(self, x, segmap, mask, attrs_matrix):
        if self.region_encoder:
            attrs_matrix = None
        x_middle, codes_vector = self.region_attr_encoder(x, segmap, attrs_matrix)
        output = self.decoder(x_middle, segmap, codes_vector, mask)
        return output


class DMFBLayer(nn.Module, ABC):
    def __init__(self, conv_cdim, dilation_rate=2):
        super(DMFBLayer, self).__init__()
        middle_cdim = conv_cdim // 4
        self.conv3 = nn.Conv2d(conv_cdim, middle_cdim, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(middle_cdim, middle_cdim, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(middle_cdim, middle_cdim, kernel_size=3, padding=2, dilation=dilation_rate)
        self.conv3_4 = nn.Conv2d(middle_cdim, middle_cdim, kernel_size=3, padding=4, dilation=dilation_rate ** 2)
        self.conv3_8 = nn.Conv2d(middle_cdim, middle_cdim, kernel_size=3, padding=8, dilation=dilation_rate ** 3)

        self.conv3_12 = nn.Conv2d(middle_cdim, middle_cdim, kernel_size=3, padding=1)
        self.conv3_24 = nn.Conv2d(middle_cdim, middle_cdim, kernel_size=3, padding=1)
        self.conv3_48 = nn.Conv2d(middle_cdim, middle_cdim, kernel_size=3, padding=1)

        self.in_norm = nn.InstanceNorm2d(middle_cdim, affine=True)

        self.conv3_concat = nn.Conv2d(conv_cdim, conv_cdim, kernel_size=1)
        self.in_norm_concat = nn.InstanceNorm2d(conv_cdim, affine=True)
        self.acf = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x1 = self.acf(self.in_norm(self.conv3(x)))

        x2_1 = self.acf(self.conv3_1(x1))
        x2_2 = self.acf(self.conv3_2(x1))
        x2_4 = self.acf(self.conv3_4(x1))
        x2_8 = self.acf(self.conv3_8(x1))

        k2 = self.conv3_12(x2_1 + x2_2)
        k3 = self.conv3_24(k2 + x2_4)
        k4 = self.conv3_48(k3 + x2_8)

        x2_concat = torch.cat([x2_1, k2, k3, k4], dim=1)
        x2 = self.in_norm_concat(self.conv3_concat(x2_concat))

        output = x2 + x
        return output


class Decoder(nn.Module, ABC):
    def __init__(self, facial_fea_names, in_nc=512, out_nc=3, add_noise=True, latent_vector_size=512,
                 region_normalized=True,
                 is_spectral_norm=True, norm_layer=nn.InstanceNorm2d):
        super(Decoder, self).__init__()

        # 4 SPADEResnetBlock
        # 0   1   2   3   4   conv
        # 512 256 128 64  32  ->   3
        # 16  32  64  128 256
        self.spaderesnetblock_0 = SPADEResnetBlock(in_nc, in_nc, facial_fea_names, add_noise, latent_vector_size,
                                                   region_normalized=region_normalized,
                                                   is_spectral_norm=is_spectral_norm, norm_layer=norm_layer)
        self.spaderesnetblock_1 = SPADEResnetBlock(in_nc, in_nc // 2, facial_fea_names, add_noise, latent_vector_size,
                                                   region_normalized=region_normalized,
                                                   is_spectral_norm=is_spectral_norm, norm_layer=norm_layer)
        self.spaderesnetblock_2 = SPADEResnetBlock(in_nc // 2, in_nc // 4, facial_fea_names, add_noise,
                                                   latent_vector_size, region_normalized=region_normalized,
                                                   is_spectral_norm=is_spectral_norm, norm_layer=norm_layer)
        self.spaderesnetblock_3 = SPADEResnetBlock(in_nc // 4, in_nc // 8, facial_fea_names, add_noise,
                                                   latent_vector_size, region_normalized=region_normalized,
                                                   is_spectral_norm=is_spectral_norm, norm_layer=norm_layer)
        self.spaderesnetblock_4 = SPADEResnetBlock(in_nc // 8, in_nc // 16, facial_fea_names, add_noise,
                                                   latent_vector_size, region_normalized=region_normalized,
                                                   is_spectral_norm=is_spectral_norm, norm_layer=norm_layer)

        self.conv_img = nn.Conv2d(in_nc // 16, out_nc, 3, stride=1, padding=1)
        # up layer
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x, segmap, coder_vector, mask):
        x = self.spaderesnetblock_0(x, segmap, coder_vector, mask)
        x = self.up(x)  # 512,32
        x = self.spaderesnetblock_1(x, segmap, coder_vector, mask)
        x = self.up(x)  # 256,64
        x = self.spaderesnetblock_2(x, segmap, coder_vector, mask)
        x = self.up(x)  # 128,128
        x = self.spaderesnetblock_3(x, segmap, coder_vector, mask)
        x = self.up(x)  # 64,256
        x = self.spaderesnetblock_4(x, segmap, coder_vector, mask)  # 32,256
        x = self.conv_img(x)
        x = torch.tanh(x)

        return x


class RegionAttrEncoder(nn.Module, ABC):
    def __init__(self, facial_fea_names, facial_fea_attr_names, facial_fea_attr_len,
                 input_nc=3, output_nc=512, ngf=32, n_downsampling=3,
                 region_encoder=True, norm_layer=nn.InstanceNorm2d):
        super(RegionAttrEncoder, self).__init__()

        self.region_encoder = region_encoder
        self.output_nc = output_nc

        self.facial_fea_names = facial_fea_names
        self.num_facial_fea_names = len(self.facial_fea_names)
        self.facial_fea_attr_len = facial_fea_attr_len
        self.facial_fea_attr_names = facial_fea_attr_names

        if not self.region_encoder:  # attr encoder
            self.max_facial_fea_attr_len = max(self.facial_fea_attr_len) + 1
            self.init_attr_fea_extractor()

        fea_extractor_layers = [nn.ReflectionPad2d(1),
                                nn.Conv2d(input_nc, ngf, kernel_size=3, padding=0),
                                norm_layer(ngf),
                                nn.LeakyReLU(0.2, False)]
        # downsample
        mult = ngf
        for i in range(n_downsampling):
            fea_extractor_layers += [nn.Conv2d(mult, mult * 2, kernel_size=3, stride=2, padding=1),
                                     norm_layer(mult * 2),
                                     nn.LeakyReLU(0.2, False)]
            mult = mult * 2

        self.fea_extractor_model = nn.Sequential(*fea_extractor_layers)
        self.multi_scale_layer = DMFBLayer(mult)
        encoder_layers = [nn.Conv2d(mult, mult * 2, kernel_size=3, stride=2, padding=1),
                          norm_layer(mult * 2),
                          nn.LeakyReLU(0.2, False)]
        self.encoder = nn.Sequential(*encoder_layers)

        # upsample
        region_attr_layers = []
        for i in range(n_downsampling - 1):
            region_attr_layers += [
                nn.ConvTranspose2d(mult, mult * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(mult * 2),
                nn.LeakyReLU(0.2, False)]
            mult = mult * 2

        region_attr_layers += [nn.ReflectionPad2d(1),
                               nn.Conv2d(mult, output_nc, kernel_size=3, padding=0),
                               nn.Tanh()]

        self.region_attr_model = nn.Sequential(*region_attr_layers)

    def init_attr_fea_extractor(self):
        for facial_fea_idx, facial_fea_name in enumerate(self.facial_fea_names):
            for facial_fea_attr_idx in range(self.facial_fea_attr_len[facial_fea_idx]):
                facial_fea_attr_layer = [nn.AdaptiveAvgPool2d((self.output_nc, 1)),
                                         nn.LeakyReLU(0.2, False)]
                setattr(self, facial_fea_name + '_' + str(facial_fea_attr_idx), nn.Sequential(*facial_fea_attr_layer))

    def forward(self, x, segmap, attrs_matrix=None):
        """
           input:b×c×h×w
           segmap:b×n_class×h1×w1
           output:b×n_class×(model(input).shape[1])
        """
        middle_codes = self.fea_extractor_model(x)
        # 通道数变为512,特征图为原来的1/2(512, 128, 128)
        codes = self.region_attr_model(middle_codes)
        encoder_codes = self.encoder(self.multi_scale_layer(middle_codes))

        segmap = F.interpolate(segmap, size=codes.size()[2:], mode='nearest')

        batch_size_codes = codes.shape[0]
        channel_codes = codes.shape[1]

        # 通道数代表语义分割的类别数(和面部特征数量相同)
        # 每个通道的feature map表示面部特征的区域(skin area, hair area, mouth area, nose area, brow area, eye area ...)
        n_segmap = segmap.shape[1]

        assert n_segmap == self.num_facial_fea_names, 'segmap channels must same number of the facial feature!'

        if self.region_encoder and attrs_matrix is None:
            codes_vector = torch.zeros((batch_size_codes, n_segmap, channel_codes), dtype=codes.dtype,
                                       device=codes.device)
            for i in range(batch_size_codes):
                for j in range(n_segmap):
                    component_mask_area = torch.sum(segmap.bool()[i, j])
                    if component_mask_area > 0:
                        codes_component_feature = codes[i].masked_select(segmap.bool()[i, j]) \
                            .reshape(channel_codes, component_mask_area).mean(1)
                        codes_vector[i][j] = codes_component_feature

        elif not self.region_encoder and attrs_matrix is not None:
            codes_vector = torch.zeros((batch_size_codes, n_segmap, self.max_facial_fea_attr_len, channel_codes),
                                       dtype=codes.dtype, device=codes.device)

            # 根据语义分割图每个类对应的**区域**在深层特征(codes)中筛选出特征,
            # 每个类筛选出来的特征存储到对应**属性**所对应的每行[每行代表一个属性]上(多余的行填充为0)
            # (Batch_size,n_class, max_facial_fea_attr, channel_codes)
            # 分别代表(每个批次的图像个数, 面部特征数量, 最大的面部特征的属性个数, 每个属性编码的维度)
            for img_idx in range(batch_size_codes):  # batch_size 每个图片
                for facial_fea_idx in range(n_segmap):  # 每个面部特征(skin, hair, mouth, nose, brow, eye ...)
                    # 面部属性的每个属性(以Skin为例,即为young, smiling, Pale_Skin, Male), 以每个面部属性个数最大的为准,
                    # 如果最大长度比当前的面部特征的属性数量大,则将剩余的全部赋值为0
                    component_mask_area = torch.sum(segmap.bool()[img_idx, facial_fea_idx])
                    if component_mask_area > 0:
                        facial_seg_component_feature = codes[img_idx].masked_select(
                            segmap.bool()[img_idx, facial_fea_idx]).reshape(channel_codes, component_mask_area)
                        codes_vector[img_idx][facial_fea_idx][0] = facial_seg_component_feature.mean(1)

                        # attr codes
                        facial_seg_component_feature = torch.unsqueeze(facial_seg_component_feature, 0)
                        for facial_fea_attr_idx in range(self.facial_fea_attr_len[facial_fea_idx]):
                            if attrs_matrix[img_idx][facial_fea_idx][facial_fea_attr_idx] == 1:
                                facial_fea_attr_model = getattr(self, self.facial_fea_names[facial_fea_idx] + '_' + str(
                                    facial_fea_attr_idx))
                                out = facial_fea_attr_model(facial_seg_component_feature)
                                codes_vector[img_idx][facial_fea_idx][facial_fea_attr_idx + 1] = torch.squeeze(out,
                                                                                                               0).t()
                    else:
                        # print('{}th segmap miss! '.format(facial_fea_idx))
                        pass
        else:
            raise ValueError('error with region encoder and attrs')
        return encoder_codes, codes_vector


class SPADEResnetBlock(nn.Module, ABC):
    def __init__(self, in_nc, out_nc, facial_fea_names, add_noise=True, latent_vector_size=512,
                 region_normalized=True, is_spectral_norm=True, norm_layer=nn.InstanceNorm2d):
        super(SPADEResnetBlock, self).__init__()
        self.region_normalized = region_normalized

        self.learned_skip = (in_nc != out_nc)
        middle_nc = min(in_nc, out_nc)

        self.conv_0 = nn.Conv2d(in_nc, middle_nc, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(middle_nc, out_nc, kernel_size=3, padding=1)

        if self.learned_skip:
            self.conv_s = nn.Conv2d(in_nc, out_nc, kernel_size=1, bias=False)

        if is_spectral_norm:
            self.conv_0 = nn.utils.spectral_norm(self.conv_0)
            self.conv_1 = nn.utils.spectral_norm(self.conv_1)
            if self.learned_skip:
                self.conv_s = nn.utils.spectral_norm(self.conv_s)

        if region_normalized:
            self.region_norm_0 = RegionNorm(x_nc=in_nc, facial_fea_names=facial_fea_names, add_noise=add_noise,
                                            latent_vector_size=latent_vector_size, norm_layer=norm_layer)
            self.region_norm_1 = RegionNorm(x_nc=middle_nc, facial_fea_names=facial_fea_names, add_noise=add_noise,
                                            latent_vector_size=latent_vector_size, norm_layer=norm_layer)
            if self.learned_skip:
                self.region_norm_s = RegionNorm(x_nc=in_nc, facial_fea_names=facial_fea_names, add_noise=add_noise,
                                                latent_vector_size=latent_vector_size, norm_layer=norm_layer)
        else:
            self.attr_mask_norm_0 = AttrMaskNorm(x_nc=in_nc, facial_fea_names=facial_fea_names, add_noise=add_noise,
                                                 latent_vector_size=latent_vector_size, norm_layer=norm_layer)
            self.attr_mask_norm_1 = AttrMaskNorm(x_nc=middle_nc, facial_fea_names=facial_fea_names, add_noise=add_noise,
                                                 latent_vector_size=latent_vector_size, norm_layer=norm_layer)
            if self.learned_skip:
                self.attr_mask_norm_s = AttrMaskNorm(x_nc=in_nc, facial_fea_names=facial_fea_names, add_noise=add_noise,
                                                     latent_vector_size=latent_vector_size, norm_layer=norm_layer)

        self.act_f = nn.LeakyReLU(2e-1, True)

    def forward(self, x, segmap, codes_vector, mask):

        x_s = self.skip_layer(x, segmap, codes_vector, mask)

        if self.region_normalized:
            # region norm 1
            x_r = self.region_norm_0(x, segmap, codes_vector, mask)
            x_r = self.conv_0(self.act_f(x_r))

            # region norm 2
            x_r = self.region_norm_1(x_r, segmap, codes_vector, mask)
            x_r = self.conv_1(self.act_f(x_r))

            out = x_s + x_r
        else:
            # region norm 1
            x_a = self.attr_mask_norm_0(x, segmap, codes_vector, mask)
            x_a = self.conv_0(self.act_f(x_a))

            # region norm 2
            x_a = self.attr_mask_norm_1(x_a, segmap, codes_vector, mask)
            x_a = self.conv_1(self.act_f(x_a))

            out = x_s + x_a
        return out

    def skip_layer(self, x, segmap, codes_vector, mask):
        if self.learned_skip and self.region_normalized:
            x_s = self.region_norm_s(x, segmap, codes_vector, mask)
            x_s = self.conv_s(x_s)
        elif self.learned_skip and not self.region_normalized:
            x_s = self.attr_mask_norm_s(x, segmap, codes_vector, mask)
            x_s = self.conv_s(x_s)
        else:
            x_s = x

        return x_s


class RegionNorm(nn.Module, ABC):
    def __init__(self, x_nc, facial_fea_names, add_noise=True, latent_vector_size=512, norm_layer=nn.InstanceNorm2d):
        super(RegionNorm, self).__init__()
        self.norm_layer = norm_layer(x_nc)
        self.latent_vector_size = latent_vector_size
        self.noise_var = nn.Parameter(torch.zeros(x_nc), requires_grad=True)
        self.facial_fea_names = facial_fea_names
        self.add_noise = add_noise

        for facial_fea_name in facial_fea_names:
            setattr(self, facial_fea_name + '_fc', nn.Linear(latent_vector_size, latent_vector_size))

        self.conv_gamma = nn.Conv2d(latent_vector_size, x_nc, kernel_size=3, padding=1)
        self.conv_beta = nn.Conv2d(latent_vector_size, x_nc, kernel_size=3, padding=1)

        # self.spade = SPADE(x_nc, label_nc=len(facial_fea_names))
        self.spade = SPADE(x_nc, label_nc=3)

        self.blending_gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.blending_beta = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x, segmap, codes_vector, mask):
        # Part 1. generate parameter-free normalized activations
        if self.add_noise:
            added_noise = (torch.randn(x.shape[0], x.shape[3], x.shape[2], 1, dtype=x.dtype, device=x.device)
                           * self.noise_var).transpose(1, 3)
            x_norm = self.norm_layer(x + added_noise)
        else:
            x_norm = self.norm_layer(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        mask = F.interpolate(mask, size=x.size()[2:], mode='nearest')

        assert len(self.facial_fea_names) == segmap.shape[1] and len(self.facial_fea_names) == codes_vector.shape[1], \
            'num_facial_fea, num_segmap_channel and num_codes_vector_rows must be equal'
        assert codes_vector.dim() == 3, 'The dimension of codes_vector does not match, the input dimension is {},' \
                                        ' but the expected dimension is 3.'.format(codes_vector.dim())
        [b_size, f_size, h_size, w_size] = x_norm.shape
        middle_avg = torch.zeros((b_size, self.latent_vector_size, h_size, w_size), device=x_norm.device)

        for i in range(b_size):
            for j, facial_fea_name in enumerate(self.facial_fea_names):
                component_mask_area = torch.sum(segmap.bool()[i, j])
                if component_mask_area > 0:
                    facial_fea_fc = getattr(self, facial_fea_name + '_fc')
                    middle_mu = F.relu(facial_fea_fc(codes_vector[i][j]))
                    component_mu = middle_mu.reshape(self.latent_vector_size, 1).expand(self.latent_vector_size,
                                                                                        component_mask_area)
                    middle_avg[i].masked_scatter_(segmap.bool()[i, j], component_mu)

        gamma_avg = self.conv_gamma(middle_avg)
        beta_avg = self.conv_beta(middle_avg)

        # gamma_spade, beta_spade = self.spade(segmap)
        gamma_spade, beta_spade = self.spade(mask)

        gamma_alpha = torch.sigmoid(self.blending_gamma)
        beta_alpha = torch.sigmoid(self.blending_beta)

        gamma_final = gamma_alpha * gamma_avg + (1 - gamma_alpha) * gamma_spade
        beta_final = beta_alpha * beta_avg + (1 - beta_alpha) * beta_spade
        out = x_norm * (1 + gamma_final) + beta_final
        return out


class SPADE(nn.Module, ABC):
    def __init__(self, out_nc, label_nc, n_hidden=128):
        super(SPADE, self).__init__()
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        self.n_hidden = n_hidden

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, n_hidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(n_hidden, out_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(n_hidden, out_nc, kernel_size=3, padding=1)

    def forward(self, segmap):
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        return gamma, beta


class AttrMaskNorm(nn.Module, ABC):
    def __init__(self, x_nc, facial_fea_names, add_noise=True, latent_vector_size=512, norm_layer=nn.InstanceNorm2d):
        super(AttrMaskNorm, self).__init__()
        self.noise_var = nn.Parameter(torch.zeros(x_nc), requires_grad=True)
        self.norm_layer = norm_layer(x_nc)
        self.add_noise = add_noise

        self.latent_vector_size = latent_vector_size
        self.facial_fea_names = facial_fea_names
        self.num_facial_fea = len(self.facial_fea_names)

        for facial_fea_name in facial_fea_names:
            facial_fea_attr_layers = [
                nn.Conv2d(1, 1, (1, 3), stride=1, padding=1, bias=False),
                norm_layer(1),
                nn.LeakyReLU(0.2, False)
            ]
            facial_fea_model = nn.Sequential(*facial_fea_attr_layers)
            setattr(self, facial_fea_name + '_channel_model', facial_fea_model)

        self.spade = SPADE(x_nc, label_nc=3)

        self.conv_gamma = nn.Conv2d(latent_vector_size, x_nc, kernel_size=3, padding=1)
        self.conv_beta = nn.Conv2d(latent_vector_size, x_nc, kernel_size=3, padding=1)

        self.blending_gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.blending_beta = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x, segmap, codes_vector, mask):
        # Part 1. generate parameter-free normalized activations
        if self.add_noise:
            added_noise = (torch.randn(x.shape[0], x.shape[3], x.shape[2], 1, dtype=x.dtype, device=x.device)
                           * self.noise_var).transpose(1, 3)
            x_normalized = self.norm_layer(x + added_noise)
        else:
            x_normalized = self.norm_layer(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        mask = F.interpolate(mask, size=x.size()[2:], mode='nearest')

        assert self.num_facial_fea == segmap.shape[1] and self.num_facial_fea == codes_vector.shape[1], \
            'num_facial_fea, num_segmap_channel and num_codes_vector_channel must be equal'
        assert codes_vector.dim() == 4, 'The dimension of codes_vector does not match, the input dimension is {},' \
                                        ' but the expected dimension is 4.'.format(codes_vector.dim())

        [b_size, c_size, h_size, w_size] = x_normalized.shape
        # 一定要新建一个矩阵存储属性和区域属性叠加后的结果,而不是将叠加后的结果在赋值到原矩阵的第一行中,会报
        # one of the variables needed for gradient computation has been modified by an inplace operation
        # Hint: enable anomaly detection to find the operation that failed to compute its gradient,
        # with torch.autograd.set_detect_anomaly(True).
        middle_area_matrix = torch.zeros((b_size, self.num_facial_fea, 1, self.latent_vector_size), device=x.device)

        area_matrix = codes_vector[:, :, 0:1, :]
        attrs_matrix = codes_vector[:, :, 1:codes_vector.shape[2], :]

        for facial_fea_idx, facial_fea_name in enumerate(self.facial_fea_names):
            facial_fea_attr_value = attrs_matrix[:, facial_fea_idx:facial_fea_idx + 1, :, :]
            attr_conv_out = getattr(self, facial_fea_name + '_channel_model')(facial_fea_attr_value)
            middle_area_matrix[:, facial_fea_idx:facial_fea_idx + 1, :, :] = \
                area_matrix[:, facial_fea_idx:facial_fea_idx + 1, :, :] + attr_conv_out.mean(dim=2, keepdim=True)

        middle_style_matrix = torch.zeros((b_size, self.latent_vector_size, h_size, w_size), device=x.device)

        assert len(self.facial_fea_names) == segmap.shape[1], 'label_nc channel must equal segmap.shape[1]'

        for i in range(b_size):
            for j in range(segmap.shape[1]):
                component_mask_area = torch.sum(segmap.bool()[i, j])
                if component_mask_area > 0:
                    component_mu = middle_area_matrix[i][j].reshape(self.latent_vector_size, 1).expand(
                        self.latent_vector_size, component_mask_area)
                    middle_style_matrix[i].masked_scatter_(segmap.bool()[i, j], component_mu)

        gamma_style = self.conv_gamma(middle_style_matrix)
        beta_style = self.conv_beta(middle_style_matrix)

        gamma_spade, beta_spade = self.spade(mask)
        # gamma_spade, beta_spade = self.spade(segmap)

        gamma_alpha = torch.sigmoid(self.blending_gamma)
        beta_alpha = torch.sigmoid(self.blending_beta)

        gamma_final = gamma_alpha * gamma_style + (1 - gamma_alpha) * gamma_spade
        beta_final = beta_alpha * beta_style + (1 - beta_alpha) * beta_spade
        out = x_normalized * (1 + gamma_final) + beta_final

        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


# Discriminator
class NLayerDiscriminator(nn.Module, ABC):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, use_spectral_norm=True):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), use_spectral_norm),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                        kernel_size=kw, stride=2, padding=padw, bias=use_bias), use_spectral_norm),

                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.num_branch_channel = ndf * nf_mult_prev
        sequence += [
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                    kernel_size=kw, stride=2, padding=padw, bias=use_bias), use_spectral_norm),

            nn.LeakyReLU(0.2, True)
        ]

        sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult, 1,
                                             kernel_size=kw, stride=2, padding=padw, bias=use_bias), use_spectral_norm)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.kw = kw
        self.padw = padw
        self.sequence = sequence

        self.model = nn.Sequential(*self.sequence)

    def forward(self, inputs, segmap=None):
        """Standard forward."""
        return self.model(inputs), segmap


class NLayerClassifierDiscriminator(NLayerDiscriminator, ABC):
    def __init__(self, facial_fea_names, facial_fea_attr_len, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, use_spectral_norm=False):
        super(NLayerClassifierDiscriminator, self).__init__(input_nc, ndf, n_layers, norm_layer, use_sigmoid,
                                                            use_spectral_norm)
        self.facial_fea_names = facial_fea_names
        self.feature_extractor = nn.Sequential(*self.sequence[:6])
        self.dis_model = nn.Sequential(*self.sequence[6:])

        facial_fea_classifer = [
            spectral_norm(nn.Conv2d(self.num_branch_channel, self.num_branch_channel * 2,
                                    kernel_size=self.kw, stride=2, padding=self.padw), use_spectral_norm),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(self.num_branch_channel * 2, self.num_branch_channel * 4,
                                    kernel_size=self.kw, stride=2, padding=self.padw), use_spectral_norm),
            nn.LeakyReLU(0.2, True)
        ]
        for idx, facial_fea_name in enumerate(facial_fea_names):
            setattr(self, facial_fea_name + '_conv', nn.Sequential(*facial_fea_classifer))
            layers_fc = [nn.Linear(self.num_branch_channel * 4, self.num_branch_channel * 2),
                         nn.Linear(self.num_branch_channel * 2, facial_fea_attr_len[idx])]
            setattr(self, facial_fea_name + '_fc', nn.Sequential(*layers_fc))

        self.avg_pool = nn.AvgPool2d(8)

    def forward(self, inputs, segmap=None):
        assert segmap is not None, "segmap can not be None"

        ex = self.feature_extractor(inputs)
        segmap = F.interpolate(segmap, size=ex.size()[2:], mode='nearest')

        dis_out = self.dis_model(ex)

        result = []
        for idx, facial_fea_name in enumerate(self.facial_fea_names):
            out_conv = getattr(self, facial_fea_name + '_conv')(ex * segmap[:, idx:idx + 1, :, :])
            out = self.avg_pool(out_conv)
            out = out.view(out.size(0), -1)
            out = getattr(self, facial_fea_name + '_fc')(out)
            result.append(out)

        return dis_out, result
