from torch import nn
from torch.nn import functional as F

from .base import GlobalConvolutionalNetwork, BoundaryRefinement, DeconvConv2dBnRelu
from .encoders import get_encoder_channel_nr


class LargeKernelMatters(nn.Module):
    """PyTorch LKM model using ResNet(18, 34, 50, 101 or 152) encoder.

        https://arxiv.org/pdf/1703.02719.pdf
    """

    def __init__(self, encoder, num_classes, kernel_size=9, internal_channels=21, use_relu=False, pool0=False,
                 dropout_2d=0.0):
        super().__init__()

        self.dropout_2d = dropout_2d
        self.pool0 = pool0

        self.encoder = encoder
        encoder_channel_nr = get_encoder_channel_nr(self.encoder)

        self.gcn2 = GlobalConvolutionalNetwork(in_channels=encoder_channel_nr[0],
                                               out_channels=internal_channels,
                                               kernel_size=kernel_size,
                                               use_relu=use_relu)
        self.gcn3 = GlobalConvolutionalNetwork(in_channels=encoder_channel_nr[1],
                                               out_channels=internal_channels,
                                               kernel_size=kernel_size,
                                               use_relu=use_relu)
        self.gcn4 = GlobalConvolutionalNetwork(in_channels=encoder_channel_nr[2],
                                               out_channels=internal_channels,
                                               kernel_size=kernel_size,
                                               use_relu=use_relu)
        self.gcn5 = GlobalConvolutionalNetwork(in_channels=encoder_channel_nr[3],
                                               out_channels=internal_channels,
                                               kernel_size=kernel_size,
                                               use_relu=use_relu)
        self.enc_br2 = BoundaryRefinement(in_channels=internal_channels,
                                          out_channels=internal_channels,
                                          kernel_size=3)

        self.enc_br3 = BoundaryRefinement(in_channels=internal_channels,
                                          out_channels=internal_channels,
                                          kernel_size=3)

        self.enc_br4 = BoundaryRefinement(in_channels=internal_channels,
                                          out_channels=internal_channels,
                                          kernel_size=3)

        self.enc_br5 = BoundaryRefinement(in_channels=internal_channels,
                                          out_channels=internal_channels,
                                          kernel_size=3)

        self.dec_br1 = BoundaryRefinement(in_channels=internal_channels,
                                          out_channels=internal_channels,
                                          kernel_size=3)

        self.dec_br2 = BoundaryRefinement(in_channels=internal_channels,
                                          out_channels=internal_channels,
                                          kernel_size=3)

        self.dec_br3 = BoundaryRefinement(in_channels=internal_channels,
                                          out_channels=internal_channels,
                                          kernel_size=3)

        self.dec_br4 = BoundaryRefinement(in_channels=internal_channels,
                                          out_channels=internal_channels,
                                          kernel_size=3)

        self.deconv5 = DeconvConv2dBnRelu(in_channels=internal_channels, out_channels=internal_channels)
        self.deconv4 = DeconvConv2dBnRelu(in_channels=internal_channels, out_channels=internal_channels)
        self.deconv3 = DeconvConv2dBnRelu(in_channels=internal_channels, out_channels=internal_channels)
        self.deconv2 = DeconvConv2dBnRelu(in_channels=internal_channels, out_channels=internal_channels)

        self.deconv1 = DeconvConv2dBnRelu(in_channels=internal_channels, out_channels=internal_channels)
        self.dec_br0_1 = BoundaryRefinement(in_channels=internal_channels,
                                            out_channels=internal_channels,
                                            kernel_size=3)
        self.dec_br0_2 = BoundaryRefinement(in_channels=internal_channels,
                                            out_channels=internal_channels,
                                            kernel_size=3)

        self.final = nn.Conv2d(internal_channels, num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        encoder2, encoder3, encoder4, encoder5 = self.encoder(x)
        encoder5 = F.dropout2d(encoder5, p=self.dropout_2d)

        gcn2 = self.enc_br2(self.gcn2(encoder2))
        gcn3 = self.enc_br3(self.gcn3(encoder3))
        gcn4 = self.enc_br4(self.gcn4(encoder4))
        gcn5 = self.enc_br5(self.gcn5(encoder5))

        decoder5 = self.deconv5(gcn5)
        decoder4 = self.deconv4(self.dec_br4(decoder5 + gcn4))
        decoder3 = self.deconv3(self.dec_br3(decoder4 + gcn3))
        decoder2 = self.dec_br1(self.deconv2(self.dec_br2(decoder3 + gcn2)))

        if self.pool0:
            decoder2 = self.dec_br0_2(self.deconv1(self.dec_br0_1(decoder2)))

        return self.final(decoder2)
