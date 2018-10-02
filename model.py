import mxnet as mx
from mxnet.gluon import nn, HybridBlock
from mxnet.gluon.model_zoo import vision as models


class UpBlock(HybridBlock):
    """ Performs UpConvolution  """

    def __init__(self, in_ch, out_ch, numlayers, stage, block, **kwargs):
        super(UpBlock, self).__init__(**kwargs)
        self.upsample = nn.Conv2DTranspose(
            out_ch, 2, 2, padding=0, in_channels=in_ch, use_bias=False)
        self.bn = nn.BatchNorm()
        self.blocks = nn.HybridSequential(prefix="stage%d_" % stage)
        with self.blocks.name_scope():
            for layer in range(numlayers):
                if(layer == 0):
                    self.blocks.add(
                        block(out_ch, 1, True, in_channels=in_ch, prefix=''))
                    continue
                self.blocks.add(
                    block(out_ch, 1, False, in_channels=out_ch, prefix=''))

    def hybrid_forward(self, F, x, y):
        x = self.upsample(x)
        x = F.Activation(self.bn(x), act_type='relu')
        x = F.concat(x, y, dim=1)
        x = self.blocks(x)
        return x


class ResUNet(HybridBlock):
    def __init__(self, layers, channels, classes, train, enctype=18, **kwargs):
        super(ResUNet, self).__init__(**kwargs)

        block = models.BasicBlockV2 if enctype < 44 else models.BottleneckV2

        resnet = models.get_resnet(2, enctype, pretrained=train, ctx=mx.gpu())

        self.intro = nn.HybridSequential(prefix='')
        with self.intro.name_scope():
            for i in range(5):
                self.intro.add(resnet.features[i])

        self.encoder = []
        for i in range(len(layers)):
            x = resnet.features[5 + i]
            self.register_child(x)
            self.encoder.append(x)

        self.obn = resnet.features[9]
        self.oact = resnet.features[10]
        with self.name_scope():
            self.decoder = []
            for i in range(len(layers) - 1):
                x = UpBlock(channels[i], channels[i + 1],
                            layers[i], 5 + i, block)
                self.register_child(x)
                self.decoder.append(x)

            self.out = nn.HybridSequential(prefix='')
            with self.out.name_scope():
                self.out.add(nn.Conv2D(channels[3], 3, 1, use_bias=False))
                self.out.add(nn.BatchNorm())
                self.out.add(nn.Activation(activation='relu'))
                self.out.add(nn.Conv2DTranspose(
                    classes * 2, 2, 2, use_bias=False))
                self.out.add(nn.BatchNorm())
                self.out.add(nn.Activation(activation='relu'))
                self.out.add(nn.Conv2DTranspose(classes, 2, 2, use_bias=False))

    def hybrid_forward(self, F, x):
        x = self.intro(x)
        outs = []
        for i in range(len(self.encoder)):
            k = self.encoder[i]
            x = k(x)
            outs.append(x)

        x = self.obn(x)
        x = self.oact(x)
        outs.pop()

        for i in range(len(self.decoder)):
            x = self.decoder[i](x, outs.pop())

        return F.log_softmax(self.out(x), axis=1)


spec = {18: ([2, 2, 2, 2], [512, 256, 128, 64]),
        34: ([3, 6, 4, 3], [512, 256, 128, 64]),
        50: ([3, 6, 4, 3], [2048, 1024, 512, 256]),
        101: ([3, 23, 4, 3], [2048, 1024, 512, 256]),
        152: ([3, 36, 8, 3], [2048, 1024, 512, 256])}


def get_resunet(enc, train=True):
    return ResUNet(spec[enc][0], spec[enc][1], 12, train, enc)
