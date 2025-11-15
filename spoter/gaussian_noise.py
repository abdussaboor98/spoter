
import tensorflow as tf


class GaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        noise = tf.random.normal(tf.shape(tensor), mean=self.mean, stddev=self.std, dtype=tensor.dtype)
        return tensor + noise

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


if __name__ == "__main__":
    pass
