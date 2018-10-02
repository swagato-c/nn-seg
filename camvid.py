from mxnet import image, nd
from mxnet.gluon.data import Dataset
import numpy as np
import matplotlib.pyplot as plt


class imutils:
    Sky = [128, 128, 128]
    Building = [128, 0, 0]
    Pole = [192, 192, 128]
    Road = [128, 64, 128]
    Pavement = [60, 40, 222]
    Tree = [128, 128, 0]
    SignSymbol = [192, 128, 128]
    Fence = [64, 64, 128]
    Car = [64, 0, 128]
    Pedestrian = [64, 64, 0]
    Bicyclist = [0, 128, 192]
    Unlabelled = [0, 0, 0]

    DSET_MEAN = [0.41189489566336, 0.4251328133025, 0.4326707089857]
    DSET_STD = [0.27413549931506, 0.28506257482912, 0.28284674400252]

    label_colours = np.array([Sky, Building, Pole, Road, Pavement,
                              Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])
    tup_colours = [tuple(c) for c in label_colours]

    def view_annotated(self, tensor, plot=True):
        temp = tensor.asnumpy()
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, 11):
            r[temp == l] = imutils.label_colours[l, 0]
            g[temp == l] = imutils.label_colours[l, 1]
            b[temp == l] = imutils.label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = (r / 255.0)  # [:,:,0]
        rgb[:, :, 1] = (g / 255.0)  # [:,:,1]
        rgb[:, :, 2] = (b / 255.0)  # [:,:,2]
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

    def decode_image(self, tensor):
        inp = tensor.asnumpy().transpose((1, 2, 0))
        mean = np.array(imutils.DSET_MEAN)
        std = np.array(imutils.DSET_STD)
        inp = std * inp + mean
        return inp

    def view_image(self, tensor):
        inp = imutils.decode_image(tensor)
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        plt.show()

    def imread(self, path, flag=1):
        with open(path, 'rb') as f:
            raw = f.read()
        k = image.imdecode(raw, flag)
        return k

        def view_sample_predictions(self, net, loader, n, ctx):
            data, label = next(iter(loader))
            data = data.as_in_context(ctx=ctx)
            label = label.as_in_context(ctx=ctx)
            output = net(data)
            pred = nd.argmax(output, axis=1)
            batch_size = data.shape[0]
            for i in range(min(n, batch_size)):
                self.view_image(data[i])
                self.view_annotated(label[i])
                self.view_annotated(pred[i])


class CamVidDataset(Dataset):
    def __init__(self, path, transform, annot=True):
        self.paths = []
        with open(path) as f:
            for line in f:
                (d, l) = line.split()
                self.paths.append((d, l))
        self._transform = transform
        self._annot = annot

    def __len__(self):
        return len(self.paths)

    def _imread(self, path, flag=1):
        with open(path, 'rb') as f:
            raw = f.read()

        return image.imdecode(raw, flag)

    def __getitem__(self, idx):
        data, label = self.paths[idx]
        img_d = self._imread(data)
        if self._annot:
            img_an = self._imread(label, flag=0)
            img_an = img_an.astype('float32')
            img_an = img_an.transpose(axes=(2, 0, 1))
            img_an = img_an.reshape(1, 1, 360, 480)
            img_an = img_an.pad(mode="constant", constant_value=0.0,
                                pad_width=(0, 0, 0, 0, 12, 12, 0, 0))
            img_an = img_an.reshape(1, 384, 480)
            img_an = img_an.transpose(axes=(1, 2, 0))
            img_an, _ = image.center_crop(img_an, (472, 376))
            img_an = img_an.transpose(axes=(2, 0, 1))

        imag_d = self._transform(img_d)
        imag_d = imag_d.astype('float32')
        imag_d = imag_d.reshape(1, 3, 360, 480)
        imag_d = imag_d.pad(mode="constant", constant_value=0.0,
                            pad_width=(0, 0, 0, 0, 12, 12, 0, 0))
        imag_d = imag_d.reshape(3, 384, 480)

        if self._annot:
            return imag_d, img_an
        return imag_d
