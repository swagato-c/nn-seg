from random import shuffle, uniform
from mxnet import nd


class JointTransform:
    """docstring for JointTransform"""

    def __init__(self, pre, post):
        super(JointTransform, self).__init__()
        self.pre = pre
        self.post = post

    def __call__(self, base, mask):
        base_channels = base.shape[2]
        joint = nd.concat(base, mask, dim=2)

        shuffle(self.pre) if self.pre is not None else None
        shuffle(self.post) if self.post is not None else None

        if self.pre is not None:
            for aug, prob in self.pre:
                p = round(uniform(0, 1), 1)
                if p <= prob:
                    joint = aug(joint)

        aug_base = joint[:, :, :base_channels]
        aug_mask = joint[:, :, base_channels:]

        if self.post is not None:
            for aug, prob in self.post:
                p = round(uniform(0, 1), 1)
                if p <= prob:
                    aug_base = aug(aug_base)

        return aug_base, aug_mask
