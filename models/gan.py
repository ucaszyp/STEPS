import torch.nn as nn
import torch

from components import BasicConv, get_norm_layer, get_non_linear


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_mode, norm_layer='instance_norm', non_linear='leaky_relu'):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_mode, norm_layer, non_linear)
        self.non_linear = get_non_linear(non_linear)

    @staticmethod
    def build_conv_block(dim, padding_mode, norm_layer, non_linear):
        """
        Construct a convolutional block.
        :param dim: (int) the number of channels in the conv layer.
        :param padding_mode: (str) the name of padding layer: reflect | replicate | zero
        :param norm_layer: (str) normalization layer
        :param non_linear:
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = [
            BasicConv(dim, dim, 3, padding=1, padding_mode=padding_mode, norm_layer=norm_layer,
                      non_linear=non_linear),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False, padding_mode=padding_mode),
            get_norm_layer(norm_layer, dim)
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return self.non_linear(out)


class GeneratorEncoder(nn.Module):
    """
    Encoder for generator
    """

    def __init__(self, in_channels, num_downsampling=2, num_blocks=9, padding_mode='reflect',
                 norm_layer='instance_norm', non_linear='leaky_relu'):
        """
        Initialize a generator encoder
        :param in_channels:
        :param num_blocks:
        :param norm_layer:
        :param padding_mode:
        :param non_linear:
        """
        super(GeneratorEncoder, self).__init__()
        ngf = 64
        # components
        blocks = [BasicConv(in_channels, ngf, 7, padding=3, padding_mode=padding_mode, norm_layer=norm_layer,
                            non_linear=non_linear)]
        for i in range(num_downsampling):  # add downsampling layers
            mult = 2 ** i
            blocks.append(BasicConv(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1, padding_mode='zeros',
                                    norm_layer=norm_layer, non_linear=non_linear))
        res_blocks = []
        mult = 2 ** num_downsampling
        for i in range(num_blocks):  # add ResNet blocks
            res_blocks.append(ResnetBlock(ngf * mult, padding_mode=padding_mode, norm_layer=norm_layer,
                                          non_linear=non_linear))
        blocks.append(nn.Sequential(*res_blocks))
        # the final channels is ngf * 2 ** num_downsampling, default 256
        self.models = nn.ModuleList(blocks)

    def forward(self, x):
        """
        Return multi-scale outputs
        :param x:
        :return:
        """
        features = [x]
        for m in self.models:
            features.append(m(features[-1]))
        return features[1:]


class GeneratorDecoder(nn.Module):
    """
    Decoder for generator
    """

    def __init__(self, out_channels: int, num_upsampling: int = 2, padding_mode: str = 'reflect',
                 norm_layer='instance_norm', non_linear='leaky_relu', output_scales=None):
        super(GeneratorDecoder, self).__init__()
        ngf = 64
        # set parameters
        self.num_upsampling = num_upsampling
        self._out_scales = [0] if output_scales is None else output_scales
        # components
        self.up_convs = nn.ModuleDict()
        self.res_convs = nn.ModuleDict()
        for i in range(num_upsampling):  # add upsampling layers
            mult = 2 ** (num_upsampling - i)
            self.up_convs[str(i)] = nn.Sequential(
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                   output_padding=1, bias=False),
                get_norm_layer(norm_layer, int(ngf * mult / 2)),
                get_non_linear(non_linear)
            )
            out_scale = num_upsampling - i - 1
            if out_scale in self._out_scales:
                self.res_convs[str(out_scale)] = nn.Conv2d(int(ngf * mult / 2), out_channels, kernel_size=3, padding=1,
                                                           padding_mode=padding_mode)

    def forward(self, features: [list, torch.Tensor], frame_idx=0):
        # get input
        if isinstance(features, list):
            x = features[-1]
        elif isinstance(features, torch.Tensor):
            x = features
        else:
            raise TypeError('Unknown input type: {}.'.format(type(features)))

        out = {}
        for i in range(self.num_upsampling):
            x = self.up_convs[str(i)](x)
            # output scale
            out_scale = self.num_upsampling - 1 - i
            # defeat
            out['defeat', frame_idx, out_scale] = x
            # res
            if out_scale in self._out_scales:
                out['res', frame_idx, out_scale] = 0.5 * torch.tanh(self.res_convs[str(out_scale)](x)) + 0.5
        return out


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer='instance_norm'):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()

        use_bias = False

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        # nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                get_norm_layer(norm_layer, ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            get_norm_layer(norm_layer, ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        # output 1 channel prediction map
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        """Standard forward."""
        return self.model(x)


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - typically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def forward(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and ground truth labels.

        Parameters:
            prediction (tensor) - - typically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        else:
            raise ValueError('Unknown gan mode: {}.'.format(self.gan_mode))
        return loss
