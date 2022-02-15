import tensorflow as tf
from .OptimizerAbstract import OptimizerAbstract


class AdamOptimizer(OptimizerAbstract):
    def get(self, config):
        return tf.train.AdamOptimizer(learning_rate=config.learning_rate)