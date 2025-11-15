
import tensorflow as tf


class AdditiveGaussianNoise:
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        noise_sample = tf.random.normal(tf.shape(tensor), mean=self.mean, stddev=self.std, dtype=tensor.dtype)
        return tensor + noise_sample

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


if __name__ == "__main__":
    pass
